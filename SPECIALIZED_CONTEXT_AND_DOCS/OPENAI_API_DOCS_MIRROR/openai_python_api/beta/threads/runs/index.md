# Runs


Types:

```python
from openai.types.beta.threads import RequiredActionFunctionToolCall, Run, RunStatus
```

Methods:

- <code title="post /threads/{thread_id}/runs">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">create</a>(thread_id, \*\*<a href="src/openai/types/beta/threads/run_create_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code title="get /threads/{thread_id}/runs/{run_id}">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">retrieve</a>(run_id, \*, thread_id) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code title="post /threads/{thread_id}/runs/{run_id}">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">update</a>(run_id, \*, thread_id, \*\*<a href="src/openai/types/beta/threads/run_update_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code title="get /threads/{thread_id}/runs">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">list</a>(thread_id, \*\*<a href="src/openai/types/beta/threads/run_list_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/run.py">SyncCursorPage[Run]</a></code>
- <code title="post /threads/{thread_id}/runs/{run_id}/cancel">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">cancel</a>(run_id, \*, thread_id) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code title="post /threads/{thread_id}/runs/{run_id}/submit_tool_outputs">client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">submit_tool_outputs</a>(run_id, \*, thread_id, \*\*<a href="src/openai/types/beta/threads/run_submit_tool_outputs_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/run.py">Run</a></code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">create_and_poll</a>(\*args) -> Run</code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">create_and_stream</a>(\*args) -> AssistantStreamManager[AssistantEventHandler] | AssistantStreamManager[AssistantEventHandlerT]</code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">poll</a>(\*args) -> Run</code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">stream</a>(\*args) -> AssistantStreamManager[AssistantEventHandler] | AssistantStreamManager[AssistantEventHandlerT]</code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">submit_tool_outputs_and_poll</a>(\*args) -> Run</code>
- <code>client.beta.threads.runs.<a href="./src/openai/resources/beta/threads/runs/runs.py">submit_tool_outputs_stream</a>(\*args) -> AssistantStreamManager[AssistantEventHandler] | AssistantStreamManager[AssistantEventHandlerT]</code>

## Subsections

- [Steps](steps/index.md)
