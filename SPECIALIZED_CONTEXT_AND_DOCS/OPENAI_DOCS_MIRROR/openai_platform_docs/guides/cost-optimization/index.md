Cost optimization - OpenAI API

===============

[](http://platform.openai.com/docs/overview)

[Docs Docs](http://platform.openai.com/docs)[API reference API](http://platform.openai.com/docs/api-reference/introduction)

Log in[Sign up](http://platform.openai.com/signup)

Search K

Get started

[Overview](http://platform.openai.com/docs/overview)

[Quickstart](http://platform.openai.com/docs/quickstart)

[Models](http://platform.openai.com/docs/models)

[Pricing](http://platform.openai.com/docs/pricing)

[Libraries](http://platform.openai.com/docs/libraries)

Core concepts

[Text generation](http://platform.openai.com/docs/guides/text)

[Images and vision](http://platform.openai.com/docs/guides/images-vision)

[Audio and speech](http://platform.openai.com/docs/guides/audio)

[Structured output](http://platform.openai.com/docs/guides/structured-outputs)

[Function calling](http://platform.openai.com/docs/guides/function-calling)

[Using GPT-5](http://platform.openai.com/docs/guides/latest-model)

[Migrate to Responses API](http://platform.openai.com/docs/guides/migrate-to-responses)

Agents

[Overview](http://platform.openai.com/docs/guides/agents)

Build agents

Deploy in your product

Optimize

[Voice agents](http://platform.openai.com/docs/guides/voice-agents)

Tools

[Using tools](http://platform.openai.com/docs/guides/tools)

[Connectors and MCP](http://platform.openai.com/docs/guides/tools-connectors-mcp)

[Web search](http://platform.openai.com/docs/guides/tools-web-search)

[Code interpreter](http://platform.openai.com/docs/guides/tools-code-interpreter)

File search and retrieval

More tools

Run and scale

[Conversation state](http://platform.openai.com/docs/guides/conversation-state)

[Background mode](http://platform.openai.com/docs/guides/background)

[Streaming](http://platform.openai.com/docs/guides/streaming-responses)

[Webhooks](http://platform.openai.com/docs/guides/webhooks)

[File inputs](http://platform.openai.com/docs/guides/pdf-files)

Prompting

Reasoning

Evaluation

[Getting started](http://platform.openai.com/docs/guides/evaluation-getting-started)

[Working with evals](http://platform.openai.com/docs/guides/evals)

[Prompt optimizer](http://platform.openai.com/docs/guides/prompt-optimizer)

[External models](http://platform.openai.com/docs/guides/external-models)

[Best practices](http://platform.openai.com/docs/guides/evaluation-best-practices)

Realtime API

[Overview](http://platform.openai.com/docs/guides/realtime)

Connect

Usage

Model optimization

[Optimization cycle](http://platform.openai.com/docs/guides/model-optimization)

Fine-tuning

[Graders](http://platform.openai.com/docs/guides/graders)

Specialized models

[Image generation](http://platform.openai.com/docs/guides/image-generation)

[Video generation](http://platform.openai.com/docs/guides/video-generation)

[Text to speech](http://platform.openai.com/docs/guides/text-to-speech)

[Speech to text](http://platform.openai.com/docs/guides/speech-to-text)

[Deep research](http://platform.openai.com/docs/guides/deep-research)

[Embeddings](http://platform.openai.com/docs/guides/embeddings)

[Moderation](http://platform.openai.com/docs/guides/moderation)

Coding agents

[Codex cloud](https://developers.openai.com/codex/cloud)

[Agent internet access](https://developers.openai.com/codex/cloud/agent-internet)

[Local shell tool](http://platform.openai.com/docs/guides/tools-local-shell)

[Codex CLI](https://developers.openai.com/codex/cli)

[Codex IDE](https://developers.openai.com/codex/ide)

[Codex changelog](https://developers.openai.com/codex/changelog)

Going live

[Production best practices](http://platform.openai.com/docs/guides/production-best-practices)

Latency optimization

Cost optimization

[Overview](http://platform.openai.com/docs/guides/cost-optimization)

[Batch](http://platform.openai.com/docs/guides/batch)

[Flex processing](http://platform.openai.com/docs/guides/flex-processing)

[Accuracy optimization](http://platform.openai.com/docs/guides/optimizing-llm-accuracy)

Safety

Specialized APIs

Assistants API

Resources

[Terms and policies](https://openai.com/policies)

[Changelog](http://platform.openai.com/docs/changelog)

[Your data](http://platform.openai.com/docs/guides/your-data)

[Rate limits](http://platform.openai.com/docs/guides/rate-limits)

[Deprecations](http://platform.openai.com/docs/deprecations)

[MCP for deep research](http://platform.openai.com/docs/mcp)

[Developer mode](http://platform.openai.com/docs/guides/developer-mode)

ChatGPT Actions

[Cookbook](https://cookbook.openai.com/)[Forum](https://community.openai.com/categories)

Cost optimization
=================

Improve your efficiency and reduce costs.

Copy page

There are several ways to reduce costs when using OpenAI models. Cost and latency are typically interconnected; reducing tokens and requests generally leads to faster processing. OpenAI's Batch API and flex processing are additional ways to lower costs.

Cost and latency
----------------

To reduce latency and cost, consider the following strategies:

*   **Reduce requests**: Limit the number of necessary requests to complete tasks.
*   **Minimize tokens**: Lower the number of input tokens and optimize for shorter model outputs.
*   **Select a smaller model**: Use models that balance reduced costs and latency with maintained accuracy.

To dive deeper into these, please refer to our guide on [latency optimization](http://platform.openai.com/docs/guides/latency-optimization).

Batch API
---------

Process jobs asynchronously. The Batch API offers a straightforward set of endpoints that allow you to collect a set of requests into a single file, kick off a batch processing job to execute these requests, query for the status of that batch while the underlying requests execute, and eventually retrieve the collected results when the batch is complete.

[Get started with the Batch API →](http://platform.openai.com/docs/guides/batch)

Flex processing
---------------

Get significantly lower costs for Chat Completions or Responses requests in exchange for slower response times and occasional resource unavailability. Ieal for non-production or lower-priority tasks such as model evaluations, data enrichment, or asynchronous workloads.

[Get started with flex processing →](http://platform.openai.com/docs/guides/flex-processing)
