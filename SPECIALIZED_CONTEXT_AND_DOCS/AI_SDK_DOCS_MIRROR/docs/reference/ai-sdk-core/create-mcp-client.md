# `experimental_createMCPClient()`

Creates a lightweight Model Context Protocol (MCP) client that connects to an MCP server. The client provides:

- **Tools**: Automatic conversion between MCP tools and AI SDK tools
- **Resources**: Methods to list, read, and discover resource templates from MCP servers
- **Prompts**: Methods to list available prompts and retrieve prompt messages
- **Elicitation**: Support for handling server requests for additional input during tool execution

It currently does not support accepting notifications from an MCP server, and custom configuration of the client.

This feature is experimental and may change or be removed in the future.

## Import

```
import { experimental_createMCPClient } from "@ai-sdk/mcp"
```

## API Signature

### Parameters

### config:

MCPClientConfig

Configuration for the MCP client.

MCPClientConfig

### transport:

TransportConfig = MCPTransport | McpSSEServerConfig

Configuration for the message transport layer.

MCPTransport

### start:

() => Promise<void>

A method that starts the transport

### send:

(message: JSONRPCMessage) => Promise<void>

A method that sends a message through the transport

### close:

() => Promise<void>

A method that closes the transport

### onclose:

() => void

A method that is called when the transport is closed

### onerror:

(error: Error) => void

A method that is called when the transport encounters an error

### onmessage:

(message: JSONRPCMessage) => void

A method that is called when the transport receives a message

MCPTransportConfig

### type:

'sse' | 'http

Use Server-Sent Events for communication

### url:

string

URL of the MCP server

### headers?:

Record<string, string>

Additional HTTP headers to be sent with requests.

### authProvider?:

OAuthClientProvider

Optional OAuth provider for authorization to access protected remote MCP servers.

### name?:

string

Client name. Defaults to "ai-sdk-mcp-client"

### onUncaughtError?:

(error: unknown) => void

Handler for uncaught errors

### capabilities?:

ClientCapabilities

Optional client capabilities to advertise during initialization. For example, set { elicitation: {} } to enable handling elicitation requests from the server.

### Returns

Returns a Promise that resolves to an `MCPClient` with the following methods:

### tools:

async (options?: {
schemas?: TOOL_SCHEMAS
}) => Promise<McpToolSet<TOOL_SCHEMAS>>

Gets the tools available from the MCP server.

options

### schemas?:

TOOL_SCHEMAS

Schema definitions for compile-time type checking. When not provided, schemas are inferred from the server.

### listResources:

async (options?: {
params?: PaginatedRequest['params'];
options?: RequestOptions;
}) => Promise<ListResourcesResult>

Lists all available resources from the MCP server.

options

### params?:

PaginatedRequest['params']

Optional pagination parameters including cursor.

### options?:

RequestOptions

Optional request options including signal and timeout.

### readResource:

async (args: {
uri: string;
options?: RequestOptions;
}) => Promise<ReadResourceResult>

Reads the contents of a specific resource by URI.

args

### uri:

string

The URI of the resource to read.

### options?:

RequestOptions

Optional request options including signal and timeout.

### listResourceTemplates:

async (options?: {
options?: RequestOptions;
}) => Promise<ListResourceTemplatesResult>

Lists all available resource templates from the MCP server.

options

### options?:

RequestOptions

Optional request options including signal and timeout.

### listPrompts:

async (options?: {
params?: PaginatedRequest['params'];
options?: RequestOptions;
}) => Promise<ListPromptsResult>

Lists available prompts from the MCP server.

options

### params?:

PaginatedRequest['params']

Optional pagination parameters including cursor.

### options?:

RequestOptions

Optional request options including signal and timeout.

### getPrompt:

async (args: {
name: string;
arguments?: Record<string, unknown>;
options?: RequestOptions;
}) => Promise<GetPromptResult>

Retrieves a prompt by name, optionally passing arguments.

args

### name:

string

Prompt name to retrieve.

### arguments?:

Record<string, unknown>

Optional arguments to fill into the prompt.

### options?:

RequestOptions

Optional request options including signal and timeout.

### onElicitationRequest:

(
schema: typeof ElicitationRequestSchema,
handler: (request: ElicitationRequest) => Promise<ElicitResult> | ElicitResult
) => void

Registers a handler for elicitation requests from the MCP server. The handler receives requests when the server needs additional input during tool execution.

parameters

### schema:

typeof ElicitationRequestSchema

The schema to validate requests against. Must be ElicitationRequestSchema.

### handler:

(request: ElicitationRequest) => Promise<ElicitResult> | ElicitResult

A function that handles the elicitation request. The request contains a message and requestedSchema. The handler must return an object with an action ("accept", "decline", or "cancel") and optionally content when accepting.

### close:

async () => void

Closes the connection to the MCP server and cleans up resources.

## Example

```typescript
import {
  experimental_createMCPClient as createMCPClient,
  generateText,
} from '@ai-sdk/mcp';
import { Experimental_StdioMCPTransport } from '@ai-sdk/mcp/mcp-stdio';
import { openai } from '@ai-sdk/openai';

let client;

try {
  client = await createMCPClient({
    transport: new Experimental_StdioMCPTransport({
      command: 'node server.js',
    }),
  });

  const tools = await client.tools();

  const response = await generateText({
    model: 'anthropic/claude-sonnet-4.5',
    tools,
    messages: [{ role: 'user', content: 'Query the data' }],
  });

  console.log(response);
} catch (error) {
  console.error('Error:', error);
} finally {
  // ensure the client is closed even if an error occurs
  if (client) {
    await client.close();
  }
}
```

## Error Handling

The client throws `MCPClientError` for:

- Client initialization failures
- Protocol version mismatches
- Missing server capabilities
- Connection failures

For tool execution, errors are propagated as `CallToolError` errors.

For unknown errors, the client exposes an `onUncaughtError` callback that can be used to manually log or handle errors that are not covered by known error types.
