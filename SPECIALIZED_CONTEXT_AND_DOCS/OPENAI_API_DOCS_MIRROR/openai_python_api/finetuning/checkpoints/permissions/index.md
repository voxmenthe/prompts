# Permissions


Types:

```python
from openai.types.fine_tuning.checkpoints import (
    PermissionCreateResponse,
    PermissionRetrieveResponse,
    PermissionDeleteResponse,
)
```

Methods:

- <code title="post /fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions">client.fine_tuning.checkpoints.permissions.<a href="./src/openai/resources/fine_tuning/checkpoints/permissions.py">create</a>(fine_tuned_model_checkpoint, \*\*<a href="src/openai/types/fine_tuning/checkpoints/permission_create_params.py">params</a>) -> <a href="./src/openai/types/fine_tuning/checkpoints/permission_create_response.py">SyncPage[PermissionCreateResponse]</a></code>
- <code title="get /fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions">client.fine_tuning.checkpoints.permissions.<a href="./src/openai/resources/fine_tuning/checkpoints/permissions.py">retrieve</a>(fine_tuned_model_checkpoint, \*\*<a href="src/openai/types/fine_tuning/checkpoints/permission_retrieve_params.py">params</a>) -> <a href="./src/openai/types/fine_tuning/checkpoints/permission_retrieve_response.py">PermissionRetrieveResponse</a></code>
- <code title="delete /fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions/{permission_id}">client.fine_tuning.checkpoints.permissions.<a href="./src/openai/resources/fine_tuning/checkpoints/permissions.py">delete</a>(permission_id, \*, fine_tuned_model_checkpoint) -> <a href="./src/openai/types/fine_tuning/checkpoints/permission_delete_response.py">PermissionDeleteResponse</a></code>
