# FileBatches


Types:

```python
from openai.types.vector_stores import VectorStoreFileBatch
```

Methods:

- <code title="post /vector_stores/{vector_store_id}/file_batches">client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">create</a>(vector_store_id, \*\*<a href="src/openai/types/vector_stores/file_batch_create_params.py">params</a>) -> <a href="./src/openai/types/vector_stores/vector_store_file_batch.py">VectorStoreFileBatch</a></code>
- <code title="get /vector_stores/{vector_store_id}/file_batches/{batch_id}">client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">retrieve</a>(batch_id, \*, vector_store_id) -> <a href="./src/openai/types/vector_stores/vector_store_file_batch.py">VectorStoreFileBatch</a></code>
- <code title="post /vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel">client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">cancel</a>(batch_id, \*, vector_store_id) -> <a href="./src/openai/types/vector_stores/vector_store_file_batch.py">VectorStoreFileBatch</a></code>
- <code title="get /vector_stores/{vector_store_id}/file_batches/{batch_id}/files">client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">list_files</a>(batch_id, \*, vector_store_id, \*\*<a href="src/openai/types/vector_stores/file_batch_list_files_params.py">params</a>) -> <a href="./src/openai/types/vector_stores/vector_store_file.py">SyncCursorPage[VectorStoreFile]</a></code>
- <code>client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">create_and_poll</a>(\*args) -> VectorStoreFileBatch</code>
- <code>client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">poll</a>(\*args) -> VectorStoreFileBatch</code>
- <code>client.vector_stores.file_batches.<a href="./src/openai/resources/vector_stores/file_batches.py">upload_and_poll</a>(\*args) -> VectorStoreFileBatch</code>
