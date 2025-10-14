# Items


Types:

```python
from openai.types.conversations import ConversationItem, ConversationItemList
```

Methods:

- <code title="post /conversations/{conversation_id}/items">client.conversations.items.<a href="./src/openai/resources/conversations/items.py">create</a>(conversation_id, \*\*<a href="src/openai/types/conversations/item_create_params.py">params</a>) -> <a href="./src/openai/types/conversations/conversation_item_list.py">ConversationItemList</a></code>
- <code title="get /conversations/{conversation_id}/items/{item_id}">client.conversations.items.<a href="./src/openai/resources/conversations/items.py">retrieve</a>(item_id, \*, conversation_id, \*\*<a href="src/openai/types/conversations/item_retrieve_params.py">params</a>) -> <a href="./src/openai/types/conversations/conversation_item.py">ConversationItem</a></code>
- <code title="get /conversations/{conversation_id}/items">client.conversations.items.<a href="./src/openai/resources/conversations/items.py">list</a>(conversation_id, \*\*<a href="src/openai/types/conversations/item_list_params.py">params</a>) -> <a href="./src/openai/types/conversations/conversation_item.py">SyncConversationCursorPage[ConversationItem]</a></code>
- <code title="delete /conversations/{conversation_id}/items/{item_id}">client.conversations.items.<a href="./src/openai/resources/conversations/items.py">delete</a>(item_id, \*, conversation_id) -> <a href="./src/openai/types/conversations/conversation.py">Conversation</a></code>
