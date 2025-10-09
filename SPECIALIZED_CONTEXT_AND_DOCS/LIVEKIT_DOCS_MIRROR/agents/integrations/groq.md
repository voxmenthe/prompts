LiveKit Docs › Partner spotlight › Groq › Overview

---

# Groq and LiveKit

> Ship lightning-fast voice AI with Groq and LiveKit Agents.

## Groq ecosystem support

[Groq](https://groq.com/) provides fast AI inference in the cloud and on-prem AI compute centers. LiveKit Agents can integrate with the following Groq services:

- **STT**: Fast and cost-effective English or multilingual transcription based on `whisper-large-v3`.
- **TTS**: Fast English and Arabic text-to-speech based on `playai-tts`.
- **LLM**: Fast inference for open models like `gpt-oss-120b` and more.

Groq LLMs are available in LiveKit Inference, with billing and integration handled automatically. The Groq plugin for LiveKit Agents also provides support for STT and TTS models.

## Getting started

Use the Voice AI quickstart to build a voice AI app with Groq. Select an STT-LLM-TTS pipeline model type and add the following components to build on Groq.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Build your first voice AI app with Groq.

Use a Groq LLM to initialize your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

# ...

# in your entrypoint function
session = AgentSession(
    llm="groq/gpt-oss-120b",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "groq/gpt-oss-120b",
    // ... tts, stt, vad, turn_detection, etc.
});

```

## LiveKit Agents overview

LiveKit Agents is an open source framework for building realtime AI apps in Python and Node.js. It supports complex voice AI [workflows](https://docs.livekit.io/agents/build/workflows.md) with multiple agents and discrete processing steps, and includes built-in load balancing.

LiveKit provides SIP support for [telephony integration](https://docs.livekit.io/agents/start/telephony.md) and full-featured [frontend SDKs](https://docs.livekit.io/agents/start/frontend.md) in multiple languages. It uses [WebRTC](https://docs.livekit.io/home/get-started/intro-to-livekit.md#what-is-webrtc) transport for end-user devices, enabling high-quality, low-latency realtime experiences. To learn more, see [LiveKit Agents](https://docs.livekit.io/agents.md).

## Additional resources

The following links provide more information on each available Groq component in LiveKit Agents.

- **[Groq STT plugin](https://docs.livekit.io/agents/models/stt/plugins/groq.md)**: LiveKit Agents plugin for Groq transcription models.

- **[Groq TTS plugin](https://docs.livekit.io/agents/models/tts/plugins/groq.md)**: LiveKit Agents plugin for Groq speech models.

- **[Groq LLM plugin](https://docs.livekit.io/agents/models/llm/plugins/groq.md)**: LiveKit Agents plugin for Groq LLM models.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/groq.md](https://docs.livekit.io/agents/integrations/groq.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).