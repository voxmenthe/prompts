<!-- Source: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview -->

Claude is capable of interacting with tools and functions, allowing you to extend Claude's capabilities to perform a wider variety of tasks.

Learn everything you need to master tool use with Claude as part of our new [courses](<https://anthropic.skilljar.com/>)! Please continue to share your ideas and suggestions using this [form](<https://forms.gle/BFnYc6iCkWoRzFgk7>).

**Guarantee schema conformance with strict tool use**

[Structured Outputs](</docs/en/build-with-claude/structured-outputs>) provides guaranteed schema validation for tool inputs. Add `strict: true` to your tool definitions to ensure Claude's tool calls always match your schema exactly—no more type mismatches or missing fields.

Perfect for production agents where invalid tool parameters would cause failures. [Learn when to use strict tool use →](</docs/en/build-with-claude/structured-outputs#when-to-use-json-outputs-vs-strict-tool-use>)

Here's an example of how to provide tools to Claude using the Messages API:
[code] 
    curl https://api.anthropic.com/v1/messages \
      -H "content-type: application/json" \
      -H "x-api-key: $ANTHROPIC_API_KEY" \
      -H "anthropic-version: 2023-06-01" \
      -d '{
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "tools": [
          {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA"
                }
              },
              "required": ["location"]
            }
          }
        ],
        "messages": [
          {
            "role": "user",
            "content": "What is the weather like in San Francisco?"
          }
        ]
      }'
[/code]

* * *

## 

How tool use works

Claude supports two types of tools:

  1. **Client tools** : Tools that execute on your systems, which include:

     * User-defined custom tools that you create and implement
     * Anthropic-defined tools like [computer use](</docs/en/agents-and-tools/tool-use/computer-use-tool>) and [text editor](</docs/en/agents-and-tools/tool-use/text-editor-tool>) that require client implementation
  2. **Server tools** : Tools that execute on Anthropic's servers, like the [web search](</docs/en/agents-and-tools/tool-use/web-search-tool>) and [web fetch](</docs/en/agents-and-tools/tool-use/web-fetch-tool>) tools. These tools must be specified in the API request but don't require implementation on your part.

Anthropic-defined tools use versioned types (e.g., `web_search_20250305`, `text_editor_20250124`) to ensure compatibility across model versions.

### 

Client tools

Integrate client tools with Claude in these steps:

  1. Provide Claude with tools and a user prompt

     * Define client tools with names, descriptions, and input schemas in your API request.
     * Include a user prompt that might require these tools, e.g., "What's the weather in San Francisco?"

  2. Claude decides to use a tool

     * Claude assesses if any tools can help with the user's query.
     * If yes, Claude constructs a properly formatted tool use request.
     * For client tools, the API response has a `stop_reason` of `tool_use`, signaling Claude's intent.

  3. Execute the tool and return results

     * Extract the tool name and input from Claude's request
     * Execute the tool code on your system
     * Return the results in a new `user` message containing a `tool_result` content block

  4. Claude uses tool result to formulate a response

     * Claude analyzes the tool results to craft its final response to the original user prompt.

Note: Steps 3 and 4 are optional. For some workflows, Claude's tool use request (step 2) might be all you need, without sending results back to Claude.

### 

Server tools

Server tools follow a different workflow:

  1. Provide Claude with tools and a user prompt

     * Server tools, like [web search](</docs/en/agents-and-tools/tool-use/web-search-tool>) and [web fetch](</docs/en/agents-and-tools/tool-use/web-fetch-tool>), have their own parameters.
     * Include a user prompt that might require these tools, e.g., "Search for the latest news about AI" or "Analyze the content at this URL."

  2. Claude executes the server tool

     * Claude assesses if a server tool can help with the user's query.
     * If yes, Claude executes the tool, and the results are automatically incorporated into Claude's response.

  3. Claude uses the server tool result to formulate a response

     * Claude analyzes the server tool results to craft its final response to the original user prompt.
     * No additional user interaction is needed for server tool execution.

* * *

## 

Tool use examples

Here are a few code examples demonstrating various tool use patterns and techniques. For brevity's sake, the tools are simple tools, and the tool descriptions are shorter than would be ideal to ensure best performance.

* * *

## 

Pricing

Tool use requests are priced based on:

  1. The total number of input tokens sent to the model (including in the `tools` parameter)
  2. The number of output tokens generated
  3. For server-side tools, additional usage-based pricing (e.g., web search charges per search performed)

Client-side tools are priced the same as any other Claude API request, while server-side tools may incur additional charges based on their specific usage.

The additional tokens from tool use come from:

  * The `tools` parameter in API requests (tool names, descriptions, and schemas)
  * `tool_use` content blocks in API requests and responses
  * `tool_result` content blocks in API requests

When you use `tools`, we also automatically include a special system prompt for the model which enables tool use. The number of tool use tokens required for each model are listed below (excluding the additional tokens listed above). Note that the table assumes at least 1 tool is provided. If no `tools` are provided, then a tool choice of `none` uses 0 additional system prompt tokens.

Model| Tool choice| Tool use system prompt token count  
---|---|---  
Claude Opus 4.5| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Opus 4.1| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Opus 4| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Sonnet 4.5| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Sonnet 4| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Sonnet 3.7 ([deprecated](</docs/en/about-claude/model-deprecations>))| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Haiku 4.5| `auto`, `none`

* * *

`any`, `tool`| 346 tokens

* * *

313 tokens  
Claude Haiku 3.5| `auto`, `none`

* * *

`any`, `tool`| 264 tokens

* * *

340 tokens  
Claude Opus 3 ([deprecated](</docs/en/about-claude/model-deprecations>))| `auto`, `none`

* * *

`any`, `tool`| 530 tokens

* * *

281 tokens  
Claude Sonnet 3| `auto`, `none`

* * *

`any`, `tool`| 159 tokens

* * *

235 tokens  
Claude Haiku 3| `auto`, `none`

* * *

`any`, `tool`| 264 tokens

* * *

340 tokens  

These token counts are added to your normal input and output tokens to calculate the total cost of a request.

Refer to our [models overview table](</docs/en/about-claude/models/overview#model-comparison-table>) for current per-model prices.

When you send a tool use prompt, just like any other API request, the response will output both input and output token counts as part of the reported `usage` metrics.

* * *

## 

Next Steps

Explore our repository of ready-to-implement tool use code examples in our cookbooks:

Calculator Tool

Customer Service Agent

JSON Extractor