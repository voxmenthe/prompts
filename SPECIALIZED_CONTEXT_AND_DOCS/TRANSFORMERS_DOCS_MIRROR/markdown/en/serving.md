# Serving

Transformer models can be efficiently deployed using libraries such as vLLM, Text Generation Inference (TGI), and others. These libraries are designed for production-grade user-facing services, and can scale to multiple servers and millions of concurrent users. Refer to [Transformers as Backend for Inference Servers](./transformers_as_backends) for usage examples.

Responses API is now supported as an experimental API! Read more about it [here](#responses-api).

Apart from that you can also serve transformer models easily using the `transformers serve` CLI. This is ideal for experimentation purposes, or to run models locally for personal and private use.

In this document, we dive into the different supported endpoints and modalities; we also cover the setup of several user interfaces that can be used on top of `transformers serve` in the following guides:

* [Jan (text and MCP user interface)](./jan.md)
* [Cursor (IDE)](./cursor.md)
* [Open WebUI (text, image, speech user interface)](./open_webui.md)
* [Tiny-Agents (text and MCP CLI tool)](./tiny_agents.md)

## Serve CLI

This section is experimental and subject to change in future versions

You can serve models of diverse modalities supported by `transformers` with the `transformers serve` CLI. It spawns a local server that offers compatibility with the OpenAI SDK, which is the *de facto* standard for LLM conversations and other related tasks. This way, you can use the server from many third party applications, or test it using the `transformers chat` CLI ([docs](conversations.md#chat-cli)).

The server supports the following REST APIs:

* `/v1/chat/completions`
* `/v1/responses`
* `/v1/audio/transcriptions`
* `/v1/models`

To launch a server, simply use the `transformers serve` CLI command:


```
transformers serve
```

The simplest way to interact with the server is through our `transformers chat` CLI


```
transformers chat localhost:8000 --model-name-or-path Qwen/Qwen3-4B
```

or by sending an HTTP request, like we’ll see below.

## Chat Completions - text-based

See below for examples for text-based requests. Both LLMs and VLMs should handle

curl

python - huggingface\_hub

python - openai


```
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 0.9, "max_tokens": 1000, "stream": true, "model": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

from which you’ll receive multiple chunks in the Completions API format


```
data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

(...)
```

## Chat Completions - VLMs

The Chat Completion API also supports images; see below for examples for text-and-image-based requests.

curl

python - huggingface\_hub

python - openai


```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
```

from which you’ll receive multiple chunks in the Completions API format


```
data: {"id":"req_0","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1753366665,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"The "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"image "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}
```

## Responses API

The Responses API is the newest addition to the supported APIs of `transformers serve`.

This API is still experimental: expect bug patches and additition of new features in the coming weeks.
If you run into any issues, please let us know and we’ll work on fixing them ASAP.

Instead of the previous `/v1/chat/completions` path, the Responses API lies behind the `/v1/responses` path.
See below for examples interacting with our Responses endpoint with `curl`, as well as the Python OpenAI client.

So far, this endpoint only supports text and therefore only LLMs. VLMs to come!

curl

python - openai


```
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "stream": true,
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
```

from which you’ll receive multiple chunks in the Responses API format


```
data: {"response":{"id":"resp_req_0","created_at":1754059817.783648,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"response","output":[],"parallel_tool_calls":false,"tool_choice":"auto","tools":[],"status":"queued","text":{"format":{"type":"text"}}},"sequence_number":0,"type":"response.created"}

data: {"response":{"id":"resp_req_0","created_at":1754059817.783648,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"response","output":[],"parallel_tool_calls":false,"tool_choice":"auto","tools":[],"status":"in_progress","text":{"format":{"type":"text"}}},"sequence_number":1,"type":"response.in_progress"}

data: {"item":{"id":"msg_req_0","content":[],"role":"assistant","status":"in_progress","type":"message"},"output_index":0,"sequence_number":2,"type":"response.output_item.added"}

data: {"content_index":0,"item_id":"msg_req_0","output_index":0,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":3,"type":"response.content_part.added"}

data: {"content_index":0,"delta":"","item_id":"msg_req_0","output_index":0,"sequence_number":4,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"Once ","item_id":"msg_req_0","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"upon ","item_id":"msg_req_0","output_index":0,"sequence_number":6,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"a ","item_id":"msg_req_0","output_index":0,"sequence_number":7,"type":"response.output_text.delta"}
```

## MCP integration

The `transformers serve` server is also an MCP client, so it can interact with MCP tools in agentic use cases. This, of course, requires the use of an LLM that is designed to use tools.

At the moment, MCP tool usage in `transformers` is limited to the `qwen` family of models.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/serving.md)
