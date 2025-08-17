LiveKit Docs › Integration guides › Speech-to-text (STT) › Speechmatics

---

# Speechmatics STT integration guide

> How to use the Speechmatics STT plugin for LiveKit Agents.

## Overview

[Speechmatics](https://www.speechmatics.com/) provides enterprise-grade speech-to-text APIs. Their advanced speech models deliver highly accurate transcriptions across diverse languages, dialects, and accents. You can use the LiveKit Speechmatics plugin with the Agents framework to build voice AI agents that provide reliable, real-time transcriptions.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[speechmatics]~=1.2"

```

### Authentication

The Speechmatics plugin requires an [API key](https://docs.speechmatics.com/introduction/authentication).

Set `SPEECHMATICS_API_KEY` in your `.env` file.

### Usage

Use Speechmatics STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import speechmatics

session = AgentSession(
   stt = speechmatics.STT(),
   # ... llm, tts, etc.
)

```

### Speaker diarization

Enable [speaker diarization](https://docs.speechmatics.com/features/diarization#speaker-diarization) by initializing the STT with `diarization="speaker"` and a `speaker_diarization_config`. You need to override the entire `transcription_config` so set the other values as needed.

```python
stt = speechmatics.STT(
   transcription_config=speechmatics.types.TranscriptionConfig(
      language="en",
      operating_point="enhanced",
      enable_partials=True,
      max_delay=0.7,
      diarization="speaker",
      speaker_diarization_config={"max_speakers": 2}, # Adjust as needed
   )
)

```

Results are available as the `speaker_id` property on the events emitted by [user_input_transcribed](https://docs.livekit.io/agents/build/events.md#user_input_transcribed):

```python
from livekit.agents import UserInputTranscribedEvent

@session.on("user_input_transcribed")
def on_user_input_transcribed(event: UserInputTranscribedEvent):
   print(f"user_input_transcribed: \"[{event.speaker_id}]: {event.transcript}\"")

```

### Parameters

This section describes the key parameters for the Speechmatics STT plugin. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/speechmatics/index.html.md#livekit.plugins.speechmatics.STT) for a complete list of all available parameters.

- **`transcription_config`** _(TranscriptionConfig)_ (optional): Configuration for the transcription service. If you override this parameter, you must provide all configuration values. The following parameters are available:

- - **`language`** _(string)_ (optional) - Default: `en`: ISO 639-1 language code. All languages are global and can understand different dialects/accents. To see the list of all supported languages, see [Supported Languages](https://docs.speechmatics.com/introduction/supported-languages).
- - **`operating_point`** _(string)_ (optional) - Default: `enhanced`: Operating point to use for the transcription per required accuracy & complexity. To learn more, see [Accuracy Reference](https://docs.speechmatics.com/features/accuracy-language-packs#accuracy).
- - **`enable_partials`** _(bool)_ (optional) - Default: `true`: Partial transcripts allow you to receive preliminary transcriptions and update as more context is available until the higher-accuracy [final transcript](https://docs.speechmatics.com/rt-api-ref#addtranscript) is returned. Partials are returned faster but without any post-processing such as formatting.
- - **`max_delay`** _(number)_ (optional) - Default: `0.7`: The delay in seconds between the end of a spoken word and returning the final transcript results.
- - **`speaker_diarization_config`** _(dict)_ (optional): Configuration for speaker diarization. The following parameters are available:
- - **`max_speakers`** _(int)_ (optional) - Default: `2`: Maximum number of speakers to detect in the audio. Valid values range from 2 to 100.
- - **`speaker_sensitivity`** _(float)_ (optional) - Default: `0.5`: Sensitivity of speaker detection between 0 and 1. Higher values increase likelihood of detecting more unique speakers.
- - **`prefer_current_speaker`** _(bool)_ (optional) - Default: `false`: When true, reduces likelihood of switching between similar sounding speakers by preferring the current speaker.

## Additional resources

The following resources provide more information about using Speechmatics with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-speechmatics/)**: The `livekit-plugins-speechmatics` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/speechmatics/index.html.md#livekit.plugins.speechmatics.STT)**: Reference for the Speechmatics STT plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-speechmatics)**: View the source or contribute to the LiveKit Speechmatics STT plugin.

- **[Speechmatics docs](https://docs.speechmatics.com/introduction/)**: Speechmatics STT docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Speechmatics STT.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/stt/speechmatics.md](https://docs.livekit.io/agents/integrations/stt/speechmatics.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).