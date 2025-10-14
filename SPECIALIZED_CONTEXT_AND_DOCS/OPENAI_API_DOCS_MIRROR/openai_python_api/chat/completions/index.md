# Completions


Types:

```python
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAllowedToolChoice,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudio,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionContentPart,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartRefusal,
    ChatCompletionContentPartText,
    ChatCompletionCustomTool,
    ChatCompletionDeleted,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionCallOption,
    ChatCompletionFunctionMessageParam,
    ChatCompletionFunctionTool,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionModality,
    ChatCompletionNamedToolChoice,
    ChatCompletionNamedToolChoiceCustom,
    ChatCompletionPredictionContent,
    ChatCompletionRole,
    ChatCompletionStoreMessage,
    ChatCompletionStreamOptions,
    ChatCompletionSystemMessageParam,
    ChatCompletionTokenLogprob,
    ChatCompletionToolUnion,
    ChatCompletionToolChoiceOption,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAllowedTools,
    ChatCompletionReasoningEffort,
)
```

Methods:

- <code title="post /chat/completions">client.chat.completions.<a href="./src/openai/resources/chat/completions/completions.py">create</a>(\*\*<a href="src/openai/types/chat/completion_create_params.py">params</a>) -> <a href="./src/openai/types/chat/chat_completion.py">ChatCompletion</a></code>
- <code title="get /chat/completions/{completion_id}">client.chat.completions.<a href="./src/openai/resources/chat/completions/completions.py">retrieve</a>(completion_id) -> <a href="./src/openai/types/chat/chat_completion.py">ChatCompletion</a></code>
- <code title="post /chat/completions/{completion_id}">client.chat.completions.<a href="./src/openai/resources/chat/completions/completions.py">update</a>(completion_id, \*\*<a href="src/openai/types/chat/completion_update_params.py">params</a>) -> <a href="./src/openai/types/chat/chat_completion.py">ChatCompletion</a></code>
- <code title="get /chat/completions">client.chat.completions.<a href="./src/openai/resources/chat/completions/completions.py">list</a>(\*\*<a href="src/openai/types/chat/completion_list_params.py">params</a>) -> <a href="./src/openai/types/chat/chat_completion.py">SyncCursorPage[ChatCompletion]</a></code>
- <code title="delete /chat/completions/{completion_id}">client.chat.completions.<a href="./src/openai/resources/chat/completions/completions.py">delete</a>(completion_id) -> <a href="./src/openai/types/chat/chat_completion_deleted.py">ChatCompletionDeleted</a></code>

## Subsections

- [Messages](messages/index.md)
