# Files


Types:

```python
from openai.types.containers import FileCreateResponse, FileRetrieveResponse, FileListResponse
```

Methods:

- <code title="post /containers/{container_id}/files">client.containers.files.<a href="./src/openai/resources/containers/files/files.py">create</a>(container_id, \*\*<a href="src/openai/types/containers/file_create_params.py">params</a>) -> <a href="./src/openai/types/containers/file_create_response.py">FileCreateResponse</a></code>
- <code title="get /containers/{container_id}/files/{file_id}">client.containers.files.<a href="./src/openai/resources/containers/files/files.py">retrieve</a>(file_id, \*, container_id) -> <a href="./src/openai/types/containers/file_retrieve_response.py">FileRetrieveResponse</a></code>
- <code title="get /containers/{container_id}/files">client.containers.files.<a href="./src/openai/resources/containers/files/files.py">list</a>(container_id, \*\*<a href="src/openai/types/containers/file_list_params.py">params</a>) -> <a href="./src/openai/types/containers/file_list_response.py">SyncCursorPage[FileListResponse]</a></code>
- <code title="delete /containers/{container_id}/files/{file_id}">client.containers.files.<a href="./src/openai/resources/containers/files/files.py">delete</a>(file_id, \*, container_id) -> None</code>

## Subsections

- [Content](content/index.md)
