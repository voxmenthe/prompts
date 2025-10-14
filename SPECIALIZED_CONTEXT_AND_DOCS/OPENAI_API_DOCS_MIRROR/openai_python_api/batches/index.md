# Batches


Types:

```python
from openai.types import Batch, BatchError, BatchRequestCounts, BatchUsage
```

Methods:

- <code title="post /batches">client.batches.<a href="./src/openai/resources/batches.py">create</a>(\*\*<a href="src/openai/types/batch_create_params.py">params</a>) -> <a href="./src/openai/types/batch.py">Batch</a></code>
- <code title="get /batches/{batch_id}">client.batches.<a href="./src/openai/resources/batches.py">retrieve</a>(batch_id) -> <a href="./src/openai/types/batch.py">Batch</a></code>
- <code title="get /batches">client.batches.<a href="./src/openai/resources/batches.py">list</a>(\*\*<a href="src/openai/types/batch_list_params.py">params</a>) -> <a href="./src/openai/types/batch.py">SyncCursorPage[Batch]</a></code>
- <code title="post /batches/{batch_id}/cancel">client.batches.<a href="./src/openai/resources/batches.py">cancel</a>(batch_id) -> <a href="./src/openai/types/batch.py">Batch</a></code>
