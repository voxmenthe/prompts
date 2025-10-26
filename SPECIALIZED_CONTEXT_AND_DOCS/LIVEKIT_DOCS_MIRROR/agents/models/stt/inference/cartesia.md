LiveKit docs › Models › Speech-to-text (STT) › LiveKit Inference › Cartesia

---

# Cartesia STT

> Reference for Cartesia STT in LiveKit Inference.

## Overview

LiveKit Inference offers transcription powered by Cartesia. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#stt).

| Model name | Model ID | Languages |
| -------- | -------- | --------- |
| Ink Whisper | `cartesia/ink-whisper` | `en`, `zh`, `de`, `es`, `ru`, `ko`, `fr`, `ja`, `pt`, `tr`, `pl`, `ca`, `nl`, `ar`, `sv`, `it`, `id`, `vi`, `he`, `uk`, `el`, `ms`, `cs`, `ro`, `da`, `hu`, `ta`, `no`, `th`, `ur`, `hr`, `bg`, `lt`, `la`, `mi`, `ml`, `cy`, `sk`, `te`, `fa`, `lv`, `bn`, `sr`, `az`, `sl`, `kn`, `et`, `mk`, `br`, `eu`, `is`, `hy`, `ne`, `mn`, `bs`, `kk`, `sq`, `sw`, `gl`, `mr`, `pa`, `si`, `km`, `sn`, `yo`, `so`, `af`, `oc`, `ka`, `be`, `tg`, `sd`, `gu`, `am`, `yi`, `lo`, `uz`, `fo`, `ht`, `ps`, `tk`, `nn`, `mt`, `sa`, `lb`, `my`, `bo`, `tl`, `mg`, `as`, `tt`, `haw`, `ln`, `ha`, `ba`, `jw`, `su`, `yue` |

## Usage

To use Cartesia, pass a descriptor with the model and language to the `stt` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="cartesia/ink-whisper:en",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    stt: "cartesia/ink-whisper:en",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `STT` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    stt=inference.STT(
        model="cartesia/ink-whisper", 
        language="en"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession, inference } from '@livekit/agents';

session = new AgentSession({
    stt: new inference.STT({ 
        model: "cartesia/ink-whisper", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model to use for the STT.

- **`language`** _(string)_ (optional): Language code for the transcription. If not set, the provider default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Cartesia STT API, including `min_volume`, and `max_silence_duration_secs`. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Additional resources

The following links provide more information about Cartesia in LiveKit Inference.

- **[Cartesia Plugin](https://docs.livekit.io/agents/models/stt/plugins/cartesia.md)**: Plugin to use your own Cartesia account instead of LiveKit Inference.

- **[Cartesia TTS models](https://docs.livekit.io/agents/models/tts/inference/cartesia.md)**: Cartesia TTS models in LiveKit Inference.

- **[Cartesia docs](https://cartesia.ai/docs)**: Cartesia's official documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/inference/cartesia.md](https://docs.livekit.io/agents/models/stt/inference/cartesia.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).