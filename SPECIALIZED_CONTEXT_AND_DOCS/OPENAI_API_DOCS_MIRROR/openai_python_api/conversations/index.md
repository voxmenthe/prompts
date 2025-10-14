# Conversations


Types:

```python
from openai.types.conversations import (
    ComputerScreenshotContent,
    Conversation,
    ConversationDeleted,
    ConversationDeletedResource,
    Message,
    SummaryTextContent,
    TextContent,
    InputTextContent,
    OutputTextContent,
    RefusalContent,
    InputImageContent,
    InputFileContent,
)
```

Methods:

- <code title="post /conversations">client.conversations.<a href="./src/openai/resources/conversations/conversations.py">create</a>(\*\*<a href="src/openai/types/conversations/conversation_create_params.py">params</a>) -> <a href="./src/openai/types/conversations/conversation.py">Conversation</a></code>
- <code title="get /conversations/{conversation_id}">client.conversations.<a href="./src/openai/resources/conversations/conversations.py">retrieve</a>(conversation_id) -> <a href="./src/openai/types/conversations/conversation.py">Conversation</a></code>
- <code title="post /conversations/{conversation_id}">client.conversations.<a href="./src/openai/resources/conversations/conversations.py">update</a>(conversation_id, \*\*<a href="src/openai/types/conversations/conversation_update_params.py">params</a>) -> <a href="./src/openai/types/conversations/conversation.py">Conversation</a></code>
- <code title="delete /conversations/{conversation_id}">client.conversations.<a href="./src/openai/resources/conversations/conversations.py">delete</a>(conversation_id) -> <a href="./src/openai/types/conversations/conversation_deleted_resource.py">ConversationDeletedResource</a></code>

## Subsections

- [Items](items/index.md)
