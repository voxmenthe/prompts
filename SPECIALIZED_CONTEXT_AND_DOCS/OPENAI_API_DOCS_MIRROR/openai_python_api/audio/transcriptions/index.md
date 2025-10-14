# Transcriptions


Types:

```python
from openai.types.audio import (
    Transcription,
    TranscriptionInclude,
    TranscriptionSegment,
    TranscriptionStreamEvent,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
    TranscriptionWord,
    TranscriptionCreateResponse,
)
```

Methods:

- <code title="post /audio/transcriptions">client.audio.transcriptions.<a href="./src/openai/resources/audio/transcriptions.py">create</a>(\*\*<a href="src/openai/types/audio/transcription_create_params.py">params</a>) -> <a href="./src/openai/types/audio/transcription_create_response.py">TranscriptionCreateResponse</a></code>
