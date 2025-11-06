LiveKit docs › Deployment & operations › Session recording & transcripts

---

# Session recording and transcripts

> Export session data in video, audio, or text format.

## Overview

There are many reasons to record or persist the sessions that occur in your app, from quality monitoring to regulatory compliance. LiveKit allows you to record the video and audio from agent sessions or save the text transcripts.

## Video or audio recording

Use the [Egress feature](https://docs.livekit.io/home/egress/overview.md) to record audio and/or video. The simplest way to do this is to start a [room composite recorder](https://docs.livekit.io/home/egress/composite-recording.md) in your agent's entrypoint. This starts recording when the agent enters the room and automatically captures all audio and video shared in the room. Recording ends when all participants leave. Recordings are stored in the cloud storage provider of your choice.

### Example

This example shows how to modify the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to record sessions. It uses Amazon S3, but you can also save files to any Amazon S3-compatible storage provider, Google Cloud Storage or Azure Blob Storage.

For additional egress examples using Google and Azure, see the [Egress examples](https://docs.livekit.io/home/egress/examples.md).

To modify the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to record sessions, add the following code. The following example assumes your AWS credentials are stored in environment variables.

** Filename: `agent.py`**

```python
from livekit import api

async def entrypoint(ctx: JobContext):
    # Add the following code to the top of your entrypoint function

    # Set up recording
    req = api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=True,
        file_outputs=[api.EncodedFileOutput(
            file_type=api.EncodedFileType.OGG,
            filepath="livekit/my-room-test.ogg",
            s3=api.S3Upload(
                bucket=os.getenv("AWS_BUCKET_NAME"),
                region=os.getenv("AWS_REGION"),
                access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            ),
        )],
    )

    lkapi = api.LiveKitAPI()
    res = await lkapi.egress.start_room_composite_egress(req)

    await lkapi.aclose()

    # .. The rest of your entrypoint code follows ...

```

** Filename: `agent.ts`**

```typescript
import {
  EgressClient,
  EncodedFileOutput,
  EncodedFileType,
  EncodingOptionsPreset
} from 'livekit-server-sdk';


// Create the egress client
const egressClient = new EgressClient(
    process.env.LIVEKIT_URL.replace("wss://", "https://"),
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET);

// Define the output for the egress
const output = new EncodedFileOutput({
    fileType: EncodedFileType.MP4,
    filepath: 'likekit/my-room-test.mp4',
    output: {
        case: 's3',
        value: {
            accessKey: process.env.AWS_ACCESS_KEY_ID,
            secret: process.env.AWS_SECRET_ACCESS_KEY,
            bucket: process.env.AWS_BUCKET_NAME,
            region: process.env.AWS_REGION,
            forcePathStyle: true,
        },
    },
});

export default defineAgent({
    // ...
    entry: async (ctx: JobContext) => {
        // .. Your entrypoint code follows ...
        
        // Add the following code after the AgentSession.start() function
        await egressClient.startRoomCompositeEgress(ctx.room.name ? ctx.room.name : 'open-room', output, {
            layout: 'grid',
            encodingOptions: EncodingOptionsPreset.H264_1080P_30,
            audioOnly: false,
        });

        // ... The rest of your entrypoint code follows ...
    },
});

```

** Filename: `package.json`**

```json
  "dependencies": {
    "livekit-server-sdk": "^2.14.0"
  }

```

## Text transcripts

Text transcripts are available in realtime via the `llm_node` or the `transcription_node` as detailed in the docs on [Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md). You can use this along with other events and callbacks to record your session and any other data you need.

Additionally, you can access the `session.history property` at any time to get the entire conversation history. Using the `add_shutdown_callback` method, you can save the conversation history to a file after the user leaves and the room closes.

For more immediate access to conversation as it happens, you can listen for related [events](https://docs.livekit.io/agents/build/events.md). A `conversation_item_added` event is emitted whenever an item is added to the chat history. The `user_input_transcribed` event is emitted whenever user input is transcribed. These results might differ from the final transcription.

### Example

This example shows how to modify the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to save the conversation history to a JSON file.

** Filename: `agent.py`**

```python
from datetime import datetime
import json

def entrypoint(ctx: JobContext):
    # Add the following code to the top, before calling ctx.connect()
    
    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        # This example writes to the temporary directory, but you can save to any location
        filename = f"/tmp/transcript_{ctx.room.name}_{current_date}.json"
        
        with open(filename, 'w') as f:
            json.dump(session.history.to_dict(), f, indent=2)
            
        print(f"Transcript for {ctx.room.name} saved to {filename}")

    ctx.add_shutdown_callback(write_transcript)

    # .. The rest of your entrypoint code follows ...

```

---


For the latest version of this document, see [https://docs.livekit.io/agents/ops/recording.md](https://docs.livekit.io/agents/ops/recording.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).