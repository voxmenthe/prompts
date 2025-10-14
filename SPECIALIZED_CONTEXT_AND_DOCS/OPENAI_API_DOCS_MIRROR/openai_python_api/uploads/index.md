# Uploads


Types:

```python
from openai.types import Upload
```

Methods:

- <code title="post /uploads">client.uploads.<a href="./src/openai/resources/uploads/uploads.py">create</a>(\*\*<a href="src/openai/types/upload_create_params.py">params</a>) -> <a href="./src/openai/types/upload.py">Upload</a></code>
- <code title="post /uploads/{upload_id}/cancel">client.uploads.<a href="./src/openai/resources/uploads/uploads.py">cancel</a>(upload_id) -> <a href="./src/openai/types/upload.py">Upload</a></code>
- <code title="post /uploads/{upload_id}/complete">client.uploads.<a href="./src/openai/resources/uploads/uploads.py">complete</a>(upload_id, \*\*<a href="src/openai/types/upload_complete_params.py">params</a>) -> <a href="./src/openai/types/upload.py">Upload</a></code>

## Subsections

- [Parts](parts/index.md)
