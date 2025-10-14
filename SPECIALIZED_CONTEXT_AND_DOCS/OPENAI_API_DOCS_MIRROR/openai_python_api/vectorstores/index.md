# VectorStores


Types:

```python
from openai.types import (
    AutoFileChunkingStrategyParam,
    FileChunkingStrategy,
    FileChunkingStrategyParam,
    OtherFileChunkingStrategyObject,
    StaticFileChunkingStrategy,
    StaticFileChunkingStrategyObject,
    StaticFileChunkingStrategyObjectParam,
    VectorStore,
    VectorStoreDeleted,
    VectorStoreSearchResponse,
)
```

Methods:

- <code title="post /vector_stores">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">create</a>(\*\*<a href="src/openai/types/vector_store_create_params.py">params</a>) -> <a href="./src/openai/types/vector_store.py">VectorStore</a></code>
- <code title="get /vector_stores/{vector_store_id}">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">retrieve</a>(vector_store_id) -> <a href="./src/openai/types/vector_store.py">VectorStore</a></code>
- <code title="post /vector_stores/{vector_store_id}">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">update</a>(vector_store_id, \*\*<a href="src/openai/types/vector_store_update_params.py">params</a>) -> <a href="./src/openai/types/vector_store.py">VectorStore</a></code>
- <code title="get /vector_stores">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">list</a>(\*\*<a href="src/openai/types/vector_store_list_params.py">params</a>) -> <a href="./src/openai/types/vector_store.py">SyncCursorPage[VectorStore]</a></code>
- <code title="delete /vector_stores/{vector_store_id}">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">delete</a>(vector_store_id) -> <a href="./src/openai/types/vector_store_deleted.py">VectorStoreDeleted</a></code>
- <code title="post /vector_stores/{vector_store_id}/search">client.vector_stores.<a href="./src/openai/resources/vector_stores/vector_stores.py">search</a>(vector_store_id, \*\*<a href="src/openai/types/vector_store_search_params.py">params</a>) -> <a href="./src/openai/types/vector_store_search_response.py">SyncPage[VectorStoreSearchResponse]</a></code>

## Subsections

- [Files](files/index.md)
- [FileBatches](filebatches/index.md)
