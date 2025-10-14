# Assistants


Types:

```python
from openai.types.beta import (
    Assistant,
    AssistantDeleted,
    AssistantStreamEvent,
    AssistantTool,
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    MessageStreamEvent,
    RunStepStreamEvent,
    RunStreamEvent,
    ThreadStreamEvent,
)
```

Methods:

- <code title="post /assistants">client.beta.assistants.<a href="./src/openai/resources/beta/assistants.py">create</a>(\*\*<a href="src/openai/types/beta/assistant_create_params.py">params</a>) -> <a href="./src/openai/types/beta/assistant.py">Assistant</a></code>
- <code title="get /assistants/{assistant_id}">client.beta.assistants.<a href="./src/openai/resources/beta/assistants.py">retrieve</a>(assistant_id) -> <a href="./src/openai/types/beta/assistant.py">Assistant</a></code>
- <code title="post /assistants/{assistant_id}">client.beta.assistants.<a href="./src/openai/resources/beta/assistants.py">update</a>(assistant_id, \*\*<a href="src/openai/types/beta/assistant_update_params.py">params</a>) -> <a href="./src/openai/types/beta/assistant.py">Assistant</a></code>
- <code title="get /assistants">client.beta.assistants.<a href="./src/openai/resources/beta/assistants.py">list</a>(\*\*<a href="src/openai/types/beta/assistant_list_params.py">params</a>) -> <a href="./src/openai/types/beta/assistant.py">SyncCursorPage[Assistant]</a></code>
- <code title="delete /assistants/{assistant_id}">client.beta.assistants.<a href="./src/openai/resources/beta/assistants.py">delete</a>(assistant_id) -> <a href="./src/openai/types/beta/assistant_deleted.py">AssistantDeleted</a></code>
