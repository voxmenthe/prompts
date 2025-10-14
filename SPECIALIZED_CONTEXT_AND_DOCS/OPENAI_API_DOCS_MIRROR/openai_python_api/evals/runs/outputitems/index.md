# OutputItems


Types:

```python
from openai.types.evals.runs import OutputItemRetrieveResponse, OutputItemListResponse
```

Methods:

- <code title="get /evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}">client.evals.runs.output_items.<a href="./src/openai/resources/evals/runs/output_items.py">retrieve</a>(output_item_id, \*, eval_id, run_id) -> <a href="./src/openai/types/evals/runs/output_item_retrieve_response.py">OutputItemRetrieveResponse</a></code>
- <code title="get /evals/{eval_id}/runs/{run_id}/output_items">client.evals.runs.output_items.<a href="./src/openai/resources/evals/runs/output_items.py">list</a>(run_id, \*, eval_id, \*\*<a href="src/openai/types/evals/runs/output_item_list_params.py">params</a>) -> <a href="./src/openai/types/evals/runs/output_item_list_response.py">SyncCursorPage[OutputItemListResponse]</a></code>
