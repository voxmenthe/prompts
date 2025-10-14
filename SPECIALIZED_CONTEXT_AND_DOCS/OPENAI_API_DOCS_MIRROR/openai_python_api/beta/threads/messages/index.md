# Messages


Types:

```python
from openai.types.beta.threads import (
    Annotation,
    AnnotationDelta,
    FileCitationAnnotation,
    FileCitationDeltaAnnotation,
    FilePathAnnotation,
    FilePathDeltaAnnotation,
    ImageFile,
    ImageFileContentBlock,
    ImageFileDelta,
    ImageFileDeltaBlock,
    ImageURL,
    ImageURLContentBlock,
    ImageURLDelta,
    ImageURLDeltaBlock,
    Message,
    MessageContent,
    MessageContentDelta,
    MessageContentPartParam,
    MessageDeleted,
    MessageDelta,
    MessageDeltaEvent,
    RefusalContentBlock,
    RefusalDeltaBlock,
    Text,
    TextContentBlock,
    TextContentBlockParam,
    TextDelta,
    TextDeltaBlock,
)
```

Methods:

- <code title="post /threads/{thread_id}/messages">client.beta.threads.messages.<a href="./src/openai/resources/beta/threads/messages.py">create</a>(thread_id, \*\*<a href="src/openai/types/beta/threads/message_create_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/message.py">Message</a></code>
- <code title="get /threads/{thread_id}/messages/{message_id}">client.beta.threads.messages.<a href="./src/openai/resources/beta/threads/messages.py">retrieve</a>(message_id, \*, thread_id) -> <a href="./src/openai/types/beta/threads/message.py">Message</a></code>
- <code title="post /threads/{thread_id}/messages/{message_id}">client.beta.threads.messages.<a href="./src/openai/resources/beta/threads/messages.py">update</a>(message_id, \*, thread_id, \*\*<a href="src/openai/types/beta/threads/message_update_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/message.py">Message</a></code>
- <code title="get /threads/{thread_id}/messages">client.beta.threads.messages.<a href="./src/openai/resources/beta/threads/messages.py">list</a>(thread_id, \*\*<a href="src/openai/types/beta/threads/message_list_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/message.py">SyncCursorPage[Message]</a></code>
- <code title="delete /threads/{thread_id}/messages/{message_id}">client.beta.threads.messages.<a href="./src/openai/resources/beta/threads/messages.py">delete</a>(message_id, \*, thread_id) -> <a href="./src/openai/types/beta/threads/message_deleted.py">MessageDeleted</a></code>
