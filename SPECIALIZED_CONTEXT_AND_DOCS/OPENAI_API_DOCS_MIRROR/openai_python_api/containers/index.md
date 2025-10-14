# Containers


Types:

```python
from openai.types import ContainerCreateResponse, ContainerRetrieveResponse, ContainerListResponse
```

Methods:

- <code title="post /containers">client.containers.<a href="./src/openai/resources/containers/containers.py">create</a>(\*\*<a href="src/openai/types/container_create_params.py">params</a>) -> <a href="./src/openai/types/container_create_response.py">ContainerCreateResponse</a></code>
- <code title="get /containers/{container_id}">client.containers.<a href="./src/openai/resources/containers/containers.py">retrieve</a>(container_id) -> <a href="./src/openai/types/container_retrieve_response.py">ContainerRetrieveResponse</a></code>
- <code title="get /containers">client.containers.<a href="./src/openai/resources/containers/containers.py">list</a>(\*\*<a href="src/openai/types/container_list_params.py">params</a>) -> <a href="./src/openai/types/container_list_response.py">SyncCursorPage[ContainerListResponse]</a></code>
- <code title="delete /containers/{container_id}">client.containers.<a href="./src/openai/resources/containers/containers.py">delete</a>(container_id) -> None</code>

## Subsections

- [Files](files/index.md)
