# Steps


Types:

```python
from openai.types.beta.threads.runs import (
    CodeInterpreterLogs,
    CodeInterpreterOutputImage,
    CodeInterpreterToolCall,
    CodeInterpreterToolCallDelta,
    FileSearchToolCall,
    FileSearchToolCallDelta,
    FunctionToolCall,
    FunctionToolCallDelta,
    MessageCreationStepDetails,
    RunStep,
    RunStepDelta,
    RunStepDeltaEvent,
    RunStepDeltaMessageDelta,
    RunStepInclude,
    ToolCall,
    ToolCallDelta,
    ToolCallDeltaObject,
    ToolCallsStepDetails,
)
```

Methods:

- <code title="get /threads/{thread_id}/runs/{run_id}/steps/{step_id}">client.beta.threads.runs.steps.<a href="./src/openai/resources/beta/threads/runs/steps.py">retrieve</a>(step_id, \*, thread_id, run_id, \*\*<a href="src/openai/types/beta/threads/runs/step_retrieve_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/runs/run_step.py">RunStep</a></code>
- <code title="get /threads/{thread_id}/runs/{run_id}/steps">client.beta.threads.runs.steps.<a href="./src/openai/resources/beta/threads/runs/steps.py">list</a>(run_id, \*, thread_id, \*\*<a href="src/openai/types/beta/threads/runs/step_list_params.py">params</a>) -> <a href="./src/openai/types/beta/threads/runs/run_step.py">SyncCursorPage[RunStep]</a></code>
