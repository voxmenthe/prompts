<!-- Source: https://docs.anthropic.com/en/api/files-create -->

# Upload File

post/v1/files

Upload File

##### Header ParametersExpand Collapse 

"anthropic-beta": optional array of [AnthropicBeta](</docs/en/api/beta#anthropic_beta>)

Optional header to specify the beta version(s) you want to use.

Accepts one of the following:

UnionMember0 = string

UnionMember1 = "message-batches-2024-09-24" or "prompt-caching-2024-07-31" or "computer-use-2024-10-22" or 16 more

Accepts one of the following:

"message-batches-2024-09-24"

"prompt-caching-2024-07-31"

"computer-use-2024-10-22"

"computer-use-2025-01-24"

"pdfs-2024-09-25"

"token-counting-2024-11-01"

"token-efficient-tools-2025-02-19"

"output-128k-2025-02-19"

"files-api-2025-04-14"

"mcp-client-2025-04-04"

"mcp-client-2025-11-20"

"dev-full-thinking-2025-05-14"

"interleaved-thinking-2025-05-14"

"code-execution-2025-05-22"

"extended-cache-ttl-2025-04-11"

"context-1m-2025-08-07"

"context-management-2025-06-27"

"model-context-window-exceeded-2025-08-26"

"skills-2025-10-02"

##### ReturnsExpand Collapse 

FileMetadata = object { id, created_at, filename, 4 more } 

id: string

Unique object identifier.

The format and length of IDs may change over time.

created_at: string

RFC 3339 datetime string representing when the file was created.

formatdate-time

filename: string

Original filename of the uploaded file.

maxLength500

minLength1

mime_type: string

MIME type of the file.

maxLength255

minLength1

size_bytes: number

Size of the file in bytes.

minimum0

type: "file"

Object type.

For files, this is always `"file"`.

Accepts one of the following:

"file"

downloadable: optional boolean

Whether the file can be downloaded.

Upload File
[code]
    curl https://api.anthropic.com/v1/files \
        -H 'Content-Type: multipart/form-data' \
        -H "X-Api-Key: $ANTHROPIC_API_KEY" \
        -F file=undefined
[/code]
[code]
    {
      "id": "id",
      "created_at": "2019-12-27T18:11:19.117Z",
      "filename": "x",
      "mime_type": "x",
      "size_bytes": 0,
      "type": "file",
      "downloadable": true
    }
[/code]

##### Returns Examples
[code]
    {
      "id": "id",
      "created_at": "2019-12-27T18:11:19.117Z",
      "filename": "x",
      "mime_type": "x",
      "size_bytes": 0,
      "type": "file",
      "downloadable": true
    }
[/code]