# Evals


Types:

```python
from openai.types import (
    EvalCustomDataSourceConfig,
    EvalStoredCompletionsDataSourceConfig,
    EvalCreateResponse,
    EvalRetrieveResponse,
    EvalUpdateResponse,
    EvalListResponse,
    EvalDeleteResponse,
)
```

Methods:

- <code title="post /evals">client.evals.<a href="./src/openai/resources/evals/evals.py">create</a>(\*\*<a href="src/openai/types/eval_create_params.py">params</a>) -> <a href="./src/openai/types/eval_create_response.py">EvalCreateResponse</a></code>
- <code title="get /evals/{eval_id}">client.evals.<a href="./src/openai/resources/evals/evals.py">retrieve</a>(eval_id) -> <a href="./src/openai/types/eval_retrieve_response.py">EvalRetrieveResponse</a></code>
- <code title="post /evals/{eval_id}">client.evals.<a href="./src/openai/resources/evals/evals.py">update</a>(eval_id, \*\*<a href="src/openai/types/eval_update_params.py">params</a>) -> <a href="./src/openai/types/eval_update_response.py">EvalUpdateResponse</a></code>
- <code title="get /evals">client.evals.<a href="./src/openai/resources/evals/evals.py">list</a>(\*\*<a href="src/openai/types/eval_list_params.py">params</a>) -> <a href="./src/openai/types/eval_list_response.py">SyncCursorPage[EvalListResponse]</a></code>
- <code title="delete /evals/{eval_id}">client.evals.<a href="./src/openai/resources/evals/evals.py">delete</a>(eval_id) -> <a href="./src/openai/types/eval_delete_response.py">EvalDeleteResponse</a></code>

## Subsections

- [Runs](runs/index.md)
