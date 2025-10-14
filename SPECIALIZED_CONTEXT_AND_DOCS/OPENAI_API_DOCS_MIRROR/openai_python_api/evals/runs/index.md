# Runs


Types:

```python
from openai.types.evals import (
    CreateEvalCompletionsRunDataSource,
    CreateEvalJSONLRunDataSource,
    EvalAPIError,
    RunCreateResponse,
    RunRetrieveResponse,
    RunListResponse,
    RunDeleteResponse,
    RunCancelResponse,
)
```

Methods:

- <code title="post /evals/{eval_id}/runs">client.evals.runs.<a href="./src/openai/resources/evals/runs/runs.py">create</a>(eval_id, \*\*<a href="src/openai/types/evals/run_create_params.py">params</a>) -> <a href="./src/openai/types/evals/run_create_response.py">RunCreateResponse</a></code>
- <code title="get /evals/{eval_id}/runs/{run_id}">client.evals.runs.<a href="./src/openai/resources/evals/runs/runs.py">retrieve</a>(run_id, \*, eval_id) -> <a href="./src/openai/types/evals/run_retrieve_response.py">RunRetrieveResponse</a></code>
- <code title="get /evals/{eval_id}/runs">client.evals.runs.<a href="./src/openai/resources/evals/runs/runs.py">list</a>(eval_id, \*\*<a href="src/openai/types/evals/run_list_params.py">params</a>) -> <a href="./src/openai/types/evals/run_list_response.py">SyncCursorPage[RunListResponse]</a></code>
- <code title="delete /evals/{eval_id}/runs/{run_id}">client.evals.runs.<a href="./src/openai/resources/evals/runs/runs.py">delete</a>(run_id, \*, eval_id) -> <a href="./src/openai/types/evals/run_delete_response.py">RunDeleteResponse</a></code>
- <code title="post /evals/{eval_id}/runs/{run_id}">client.evals.runs.<a href="./src/openai/resources/evals/runs/runs.py">cancel</a>(run_id, \*, eval_id) -> <a href="./src/openai/types/evals/run_cancel_response.py">RunCancelResponse</a></code>

## Subsections

- [OutputItems](outputitems/index.md)
