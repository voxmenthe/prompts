<!-- Source: https://docs.anthropic.com/en/api/deleting-message-batches -->

# Delete a Message Batch

delete/v1/messages/batches/{message_batch_id}

Delete a Message Batch.

Message Batches can only be deleted once they've finished processing. If you'd like to delete an in-progress batch, you must first cancel it.

Learn more about the Message Batches API in our [user guide](<https://docs.claude.com/en/docs/build-with-claude/batch-processing>)

##### Path ParametersExpand Collapse 

message_batch_id: string

ID of the Message Batch.

##### ReturnsExpand Collapse 

DeletedMessageBatch = object { id, type } 

id: string

ID of the Message Batch.

type: "message_batch_deleted"

Deleted object type.

For Message Batches, this is always `"message_batch_deleted"`.

Accepts one of the following:

"message_batch_deleted"

Delete a Message Batch
[code]
    curl https://api.anthropic.com/v1/messages/batches/$MESSAGE_BATCH_ID \
        -X DELETE \
        -H "X-Api-Key: $ANTHROPIC_API_KEY"
[/code]
[code]
    {
      "id": "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF",
      "type": "message_batch_deleted"
    }
[/code]

##### Returns Examples
[code]
    {
      "id": "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF",
      "type": "message_batch_deleted"
    }
[/code]