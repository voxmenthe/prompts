# Threads


Types:

```python
from openai.types.beta import (
    AssistantResponseFormatOption,
    AssistantToolChoice,
    AssistantToolChoiceFunction,
    AssistantToolChoiceOption,
    Thread,
    ThreadDeleted,
)
```

Methods:

- <code title="post /threads">client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">create</a>(\*\*<a href="src/openai/types/beta/thread_create_params.py">params</a>) -> <a href="./src/openai/types/beta/thread.py">Thread</a></code>
- <code title="get /threads/{thread_id}">client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">retrieve</a>(thread_id) -> <a href="./src/openai/types/beta/thread.py">Thread</a></code>
- <code title="post /threads/{thread_id}">client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">update</a>(thread_id, \*\*<a href="src/openai/types/beta/thread_update_params.py">params</a>) -> <a href="./src/openai/types/beta/thread.py">Thread</a></code>
- <code title="delete /threads/{thread_id}">client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">delete</a>(thread_id) -> <a href="./src/openai/types/beta/thread_deleted.py">ThreadDeleted</a></code>
- <code title="post /threads/runs">client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">create_and_run</a>(\*\*<a href="src/openai/types/beta/thread_create_and_run_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code>client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">create_and_run_poll</a>(\*args) -> Run</code>
- <code>client.beta.threads.<a href="./src/openai/resources/beta/threads/threads.py">create_and_run_stream</a>(\*args) -> AssistantStreamManager[AssistantEventHandler] | AssistantStreamManager[AssistantEventHandlerT]</code>

## Subsections

- [Runs](runs/index.md)
- [Messages](messages/index.md)
