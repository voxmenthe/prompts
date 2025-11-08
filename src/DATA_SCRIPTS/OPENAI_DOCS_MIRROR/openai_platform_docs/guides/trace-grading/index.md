Trace grading - OpenAI API

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

[Agent evals](http://platform.openai.com/docs/guides/agent-evals)

[Trace grading](http://platform.openai.com/docs/guides/trace-grading)

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

Trace grading
=============

Grade model outputs with reproducible evaluations.

Copy page

Trace grading is the process of assigning structured scores or labels to an agent's trace—the end-to-end log of decisions, tool calls, and reasoning steps—to assess correctness, quality, or adherence to expectations. These annotations help identify where the agent did well or made mistakes, enabling targeted improvements in orchestration or behavior.

Trace evals use those graded traces to systematically evaluate agent performance across many examples, helping to benchmark changes, identify regressions, or validate improvements. Unlike black-box evaluations, trace evals provide more data to better understand why an agent succeeds or fails.

Use both features to track, analyze, and optimize the performance of groups of agents.

Get started with traces
-----------------------

1.   In the dashboard, navigate to Logs >[Traces](http://platform.openai.com/logs?api=traces).
2.   Select a worfklow. You'll see logs from any workflows you created in [Agent Builder](http://platform.openai.com/docs/guides/agent-builder).
3.   Select a trace to inspect your workflow.
4.   Create a grader, and run it to grade your agents' performance against grader criteria.

Trace grading is a valuable tool for error identification at scale, which is critical for building resilience into your AI applications. Learn more about our recommended process in our [cookbook](https://cookbook.openai.com/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md#).

Evaluate traces with runs
-------------------------

1.   Select **Grade all**. This takes you to the evaluation dashboard.
2.   In the evaluation dashboard, add and edit test criteria.
3.   Add a run to evaluate outputs. You can configure run options like model, date range, and tool calls to get more specificity in your eval.

Learn more about how you can use evals [here](http://platform.openai.com/docs/guides/evals).

*   [Overview Overview](http://platform.openai.com/docs/guides/trace-grading?timeout=30#page-top)
*   [Get started with traces Get started with traces](http://platform.openai.com/docs/guides/trace-grading?timeout=30#set-up-trace-grading)
*   [Evaluate traces with runs Evaluate traces with runs](http://platform.openai.com/docs/guides/trace-grading?timeout=30#evaluate-traces-with-runs)
