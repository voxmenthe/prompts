LiveKit docs › Models › Text-to-speech (TTS) › LiveKit Inference › Cartesia

---

# Cartesia TTS

> Reference for Cartesia TTS in LiveKit Inference.

## Overview

LiveKit Inference offers voice models powered by Cartesia. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#tts).

| Model ID | Languages |
| -------- | --------- |
| `cartesia/sonic` | `en`, `fr`, `de`, `es`, `pt`, `zh`, `ja`, `hi`, `it`, `ko`, `nl`, `pl`, `ru`, `sv`, `tr` |
| `cartesia/sonic-2` | `en`, `fr`, `de`, `es`, `pt`, `zh`, `ja`, `hi`, `it`, `ko`, `nl`, `pl`, `ru`, `sv`, `tr` |
| `cartesia/sonic-turbo` | `en`, `fr`, `de`, `es`, `pt`, `zh`, `ja`, `hi`, `it`, `ko`, `nl`, `pl`, `ru`, `sv`, `tr` |

## Usage

To use Cartesia, pass a descriptor with the model and voice to the `tts` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    # ... llm, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `TTS` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    tts=inference.TTS(
        model="cartesia/sonic-2", 
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", 
        language="en"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: new inference.TTS({ 
        model: "cartesia/sonic-2", 
        voice: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`voice`** _(string)_: See [voices](#voices) for guidance on selecting a voice.

- **`language`** _(string)_ (optional): Language code for the input text. If not set, the model default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Cartesia TTS API, including `duration`, and `speed`. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Voices

LiveKit Inference supports all of the default "Cartesia Voices" available in the Cartesia API. You can explore the available voices in the [Cartesia voice library](https://play.cartesia.ai/voices) (free account required), and use the voice by copying its ID into your LiveKit agent session.

> ℹ️ **Custom voices unavailable**
> 
> Custom Cartesia voices, including voice cloning, are not yet supported in LiveKit Inference. To use custom voices, create your own Cartesia account and use the [Cartesia plugin](https://docs.livekit.io/agents/models/tts/plugins/cartesia.md) for LiveKit Agents instead.

The following is a small sample of the Cartesia voices available in LiveKit Inference.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| Cartesia | Blake | Energetic American adult male | `en-US` | `cartesia/sonic-2:a167e0f3-df7e-4d52-a9c3-f949145efdab` |
| Cartesia | Daniela | Calm and trusting Mexican female | `es-MX` | `cartesia/sonic-2:5c5ad5e7-1020-476b-8b91-fdcbe9cc313c` |
| Cartesia | Jacqueline | Confident, young American adult female | `en-US` | `cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` |
| Cartesia | Robyn | Neutral, mature Australian female | `en-AU` | `cartesia/sonic-2:f31cc6a7-c1e8-4764-980c-60a361443dd1` |

## Additional resources

The following links provide more information about Cartesia in LiveKit Inference.

- **[Cartesia Plugin](https://docs.livekit.io/agents/models/tts/plugins/cartesia.md)**: Plugin to use your own Cartesia account instead of LiveKit Inference.

- **[Cartesia docs](https://cartesia.ai/docs)**: Cartesia's official API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/inference/cartesia.md](https://docs.livekit.io/agents/models/tts/inference/cartesia.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).