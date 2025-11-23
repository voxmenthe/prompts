LiveKit docs › Getting started › Agent builder

---

# Agent Builder

> Prototype simple voice agents directly in your browser.

## Overview

The LiveKit Agent Builder lets you prototype and deploy simple voice agents through your browser, without writing any code. It's a great way to build a proof-of-concept, explore ideas, or stand up a working prototype quickly.

The agent builder produces best-practice Python code using the LiveKit Agents SDK, and deploys your agents directly to LiveKit Cloud. The result is an agent that is fully compatible with the rest of LiveKit Cloud, including [LiveKit Inference](https://docs.livekit.io/agents/models.md#inference), and [agent insights](https://docs.livekit.io/agents/observability.md), and [agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md). You can continue iterating your agent in the builder, or convert it to code at any time to refine its behavior using [SDK-only features](#limitations).

Access the agent builder by selecting **Deploy new agent** in your project's [Agents dashboard](https://cloud.livekit.io/projects/p_/agents).

[Video: LiveKit Agents Builder](https://www.youtube.com/watch?v=FerHhAVELto)

## Agent features

The following provides a short overview of the features available to agents built in the agent builder.

### Agent name

The agent name is used for [explicit agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md#explicit). Be careful if you change the name after deploying your agent, as it may break existing dispatch rules and frontends.

### Instructions

This is the most important component of any agent. You can write a single prompt for your agent, to control its identity and behavior. See the [prompting guide](https://docs.livekit.io/agents/build/prompting.md) for tips on how to write a good prompt. You can use [variables](#variables) to include dynamic information in your prompt.

### Welcome greeting

You can choose if your agent should greet the user when they join the call, or not. If you choose to have the agent greet the user, you can also write custom instructions for the greeting. The greeting also supports [variables](#variables) for dynamic content.

### Models

Your agents support most of the models available in [LiveKit Inference](https://docs.livekit.io/agents/models.md#inference) to construct a high-performance STT-LLM-TTS pipeline. Consult the documentation on [Speech-to-text](https://docs.livekit.io/agents/models/stt.md), [Large language models](https://docs.livekit.io/agents/models/llm.md), and [Text-to-speech](https://docs.livekit.io/agents/models/tts.md) for more details on support models and voices.

### Actions

Extend your agent's functionality with HTTP tools, which call external APIs and services. HTTP tools support the following features:

- HTTP Method: GET, POST, PUT, DELETE, PATCH
- Endpoint URL: The endpoint to call, with optional path parameters using a colon prefix, for example `:user_id`
- Parameters: Query parameters (GET) or JSON body (POST, PUT, DELETE, PATCH), with optional type and description.
- Headers: Optional HTTP headers for authentication or other purposes, with support for [secrets](#secrets) and [metadata](#variables).

### Variables and metadata

Your agents automatically parse [Job metadata](https://docs.livekit.io/agents/server/job.md#metadata) as JSON and make the values available as variables in fields such as the instructions and welcome greeting. To add mock values for testing, and to add hints to the editor interface, define the metadata you intend to pass in the **Advanced** tab in the agent builder.

For instance, you can add a metadata field called `user_name`. When you dispatch the agent, include JSON `{"user_name": "<value>"}` in the metadata field, populated by your frontend app. The agent can access this value in instructions or greeting using `{{metadata.user_name}}`.

### Secrets

Secrets are secure variables that can store sensitive information like API keys, database credentials, and authentication tokens. The agent builder uses the same [secrets store](https://docs.livekit.io/agents/ops/deployment/secrets.md) as other LiveKit Cloud agents, and you can manage secrets in the same way.

Secrets are available as [variables](#variables) inside tool header values.  For instance, if you have set a secret called `ACCESS_TOKEN`, then you can use add a tool header with the name `Authorization` and value `Bearer {{secrets.ACCESS_TOKEN}}`.

### Other features

Your agent is built to use the following features, which are recommended for all voice agents built with LiveKit:

- [Background voice cancellation](https://docs.livekit.io/home/cloud/noise-cancellation.md) to improve agent comprehensision and reduce false interruptions.
- [Preemptive generation](https://docs.livekit.io/agents/build/speech.md#preemptive-generation) to improve agent responsiveness and reduce latency.
- [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector.md) for best-in-class conversational behavior

## Agent preview

The agent builder includes a live preview mode to talk to your agent as you work on it. This is a great way to quickly test your agent's behavior and iterate on your prompt or try different models and voices. Changes made in the builder are automatically applied to the preview agent.

Sessions with the preview agent use your own project's LiveKit Inference credits, but do not otherwise count against LiveKit Cloud usage. They also do not appear in [Agent observability](https://docs.livekit.io/agents/observability.md) for your project.

## Deploying to production

To deploy your agent to production, click the **Deploy agent** button in the top right corner of the builder. Your agent is now deployed just like any other LiveKit Cloud agent. See the guides on [custom frontends](https://docs.livekit.io/agents/start/frontend.md) and [telephony integrations](https://docs.livekit.io/agents/start/telephony.md) for more information on how to connect your agent to your users.

## Test frontend

After your agent is deployed to production, you can test it in a frontend built on the LiveKit Cloud [Sandbox](https://docs.livekit.io/home/cloud/sandbox.md) by clicking **Test Agent** in the top right corner of the builder. If you do not have this option, choose **Regenerate test app** from the dropdown menu to make it available.

This test frontend is a public URL that you can share with others to test your agent. More configuration for the test frontend is available in your project's [Sandbox settings](https://cloud.livekit.io/projects/p_/sandbox).

## Observing production sessions

After deploying your agent, you can observe production sessions in the [Agent insights](https://docs.livekit.io/agents/observability.md) tab in your [project's sessions dashboard](https://cloud.livekit.io/projects/p_/sessions).

## Convert to code

At any time, you can convert your agent to code by choosing the **Download code** button in the top right corner of the builder. This downloads a ZIP file containing a complete Python agent project, ready to [deploy with the LiveKit CLI](https://docs.livekit.io/agents/ops/deployment.md). Once you have deployed the new agent, you should delete the old agent in the builder so it stops receiving requests.

The generated project includes a helpful README as well as an AGENTS.md file that includes best-practices and an integration with the [LiveKit Docs MCP Server](https://docs.livekit.io/home/get-started/mcp-server.md) so that you can code in confidence with expert help from the coding assistant of your choice.

## Limitations

The agent builder is not intended to replace the LiveKit Agents SDK, but instead to make it easier to get started with voice agents which can be extended with custom code later after a proof-of-concept. The following are some of the agents SDK features that are not currently supported in the builder:

- [Workflows](https://docs.livekit.io/agents/build/workflows.md), including [handoffs](https://docs.livekit.io/agents/build/agents-handoffs.md), and [tasks](https://docs.livekit.io/agents/build/tasks.md)
- [Virtual avatars](https://docs.livekit.io/agents/models/avatars.md)
- [Vision](https://docs.livekit.io/agents/build/vision.md)
- [Realtime models](https://docs.livekit.io/agents/models/realtime.md) and [model plugins](https://docs.livekit.io/agents/models.md#plugins)
- [Tests](https://docs.livekit.io/agents/build/tests.md)

## Billing and limits

The agent builder is subject to the same [quotas and limits](https://docs.livekit.io/home/cloud/quotas-and-limits.md) as any other agent deployed to LiveKit Cloud. There is no additional cost to use the agent builder.

---


For the latest version of this document, see [https://docs.livekit.io/agents/start/builder.md](https://docs.livekit.io/agents/start/builder.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).