# Videos


Types:

```python
from openai.types import (
    Video,
    VideoCreateError,
    VideoModel,
    VideoSeconds,
    VideoSize,
    VideoDeleteResponse,
)
```

Methods:

- <code title="post /videos">client.videos.<a href="./src/openai/resources/videos.py">create</a>(\*\*<a href="src/openai/types/video_create_params.py">params</a>) -> <a href="./src/openai/types/video.py">Video</a></code>
- <code title="get /videos/{video_id}">client.videos.<a href="./src/openai/resources/videos.py">retrieve</a>(video_id) -> <a href="./src/openai/types/video.py">Video</a></code>
- <code title="get /videos">client.videos.<a href="./src/openai/resources/videos.py">list</a>(\*\*<a href="src/openai/types/video_list_params.py">params</a>) -> <a href="./src/openai/types/video.py">SyncConversationCursorPage[Video]</a></code>
- <code title="delete /videos/{video_id}">client.videos.<a href="./src/openai/resources/videos.py">delete</a>(video_id) -> <a href="./src/openai/types/video_delete_response.py">VideoDeleteResponse</a></code>
- <code title="get /videos/{video_id}/content">client.videos.<a href="./src/openai/resources/videos.py">download_content</a>(video_id, \*\*<a href="src/openai/types/video_download_content_params.py">params</a>) -> HttpxBinaryResponseContent</code>
- <code title="post /videos/{video_id}/remix">client.videos.<a href="./src/openai/resources/videos.py">remix</a>(video_id, \*\*<a href="src/openai/types/video_remix_params.py">params</a>) -> <a href="./src/openai/types/video.py">Video</a></code>
- <code>client.videos.<a href="./src/openai/resources/videos.py">create_and_poll</a>(\*args) -> Video</code>
