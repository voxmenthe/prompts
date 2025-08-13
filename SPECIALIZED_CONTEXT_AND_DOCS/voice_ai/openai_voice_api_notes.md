# OpenAI Voice API Documentation Notes

## Realtime API Overview

The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling through a WebSocket connection.

### Key Details

- **WebSocket URL**: `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01`
- **Model**: `gpt-4o-realtime-preview`
- **Supported Modalities**: Text and Audio (both input and output)
- **Protocol**: Bidirectional events protocol over WebSocket
- **Latency**: ~500ms time-to-first-byte for US clients

### Python SDK Support

The official OpenAI Python library supports the Realtime API through an async client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
    # Handle events
    async for event in connection:
        if event.type == 'error':
            print(event.error.type)
            print(event.error.code)
            print(event.error.event_id)
            print(event.error.message)
```

### Event Types

The Realtime API works through client-sent and server-sent events:

#### Client Events
- Update session configuration
- Send text and audio inputs
- Create conversation items
- Trigger responses

#### Server Events
- Confirm audio response completion
- Send text responses from model
- Error notifications
- Status updates

### Session Configuration

Sessions can be configured with:
- **Modalities**: ["text", "audio"]
- **Voice**: Options include "alloy", "shimmer", "echo"
- **Instructions**: System prompt for the assistant
- **Tools**: Function definitions for function calling
- **Turn Detection**: Voice activity detection settings

### Audio Format

- **Input**: 16kHz mono PCM audio (base64 encoded)
- **Output**: 24kHz mono PCM audio (base64 encoded)

### Cost Considerations

- **Input**: $0.06/minute
- **Output**: $0.24/minute
- Average interaction might cost ~$0.25

### Important Implementation Notes

1. **WebSocket vs WebRTC**: For production client-server apps with latency requirements, use WebRTC to connect client to server, then WebSocket from server to OpenAI
2. **State Management**: The API is stateful - maintain conversation context across the WebSocket connection
3. **Error Handling**: Implement reconnection logic with exponential backoff
4. **Audio Processing**: Allow ~300ms for audio processing and phrase endpointing to achieve 800ms total voice-to-voice latency

### References

- Official Documentation: https://platform.openai.com/docs/guides/realtime
- Model Information: https://platform.openai.com/docs/models/gpt-4o-realtime-preview
- GitHub Python SDK: https://github.com/openai/openai-python
- Pipecat Example: https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/19-openai-realtime-beta.py

### Azure OpenAI Support

Azure OpenAI also supports the Realtime API with API version `2025-04-01-preview`:

```python
from azure.openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    credential=credential,
    api_version="2025-04-01-preview"
)
```