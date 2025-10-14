# Files


Types:

```python
from openai.types.vector_stores import VectorStoreFile, VectorStoreFileDeleted, FileContentResponse
```

Methods:

- <code title="post /vector_stores/{vector_store_id}/files">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">create</a>(vector_store_id, \*\*<a href="src/openai/types/vector_stores/file_create_params.py">params</a>) -> <a href="./src/openai/types/vector_stores/vector_store_file.py">VectorStoreFile</a></code>
- <code title="get /vector_stores/{vector_store_id}/files/{file_id}">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">retrieve</a>(file_id, \*, vector_store_id) -> <a href="./src/openai/types/vector_stores/vector_store_file.py">VectorStoreFile</a></code>
- <code title="post /vector_stores/{vector_store_id}/files/{file_id}">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">update</a>(file_id, \*, vector_store_id, \*\*<a href="src/openai/types/vector_stores/file_update_params.py">params</a>) -> <a href="./src/openai/types/vector_stores/vector_store_file.py">VectorStoreFile</a></code>
- <code title="get /vector_stores/{vector_store_id}/files">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">list</a>(vector_store_id, \*\*<a href="src/openai/types/vector_stores/file_list_params.py">params</a>) -> <a href="./src/openai/types/vector_stores/vector_store_file.py">SyncCursorPage[VectorStoreFile]</a></code>
- <code title="delete /vector_stores/{vector_store_id}/files/{file_id}">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">delete</a>(file_id, \*, vector_store_id) -> <a href="./src/openai/types/vector_stores/vector_store_file_deleted.py">VectorStoreFileDeleted</a></code>
- <code title="get /vector_stores/{vector_store_id}/files/{file_id}/content">client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">content</a>(file_id, \*, vector_store_id) -> <a href="./src/openai/types/vector_stores/file_content_response.py">SyncPage[FileContentResponse]</a></code>
- <code>client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">create_and_poll</a>(\*args) -> VectorStoreFile</code>
- <code>client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">poll</a>(\*args) -> VectorStoreFile</code>
- <code>client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">upload</a>(\*args) -> VectorStoreFile</code>
- <code>client.vector_stores.files.<a href="./src/openai/resources/vector_stores/files.py">upload_and_poll</a>(\*args) -> VectorStoreFile</code>
