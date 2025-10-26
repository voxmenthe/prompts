LiveKit docs › Models › Large language models (LLM) › Plugins › LangChain

---

# LangChain integration guide

> How to use LangGraph workflows with LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [LangGraph](https://www.langchain.com/langgraph) as an LLM provider for your voice agents.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the LiveKit LangChain plugin from PyPI:

```shell
uv add "livekit-agents[langchain]~=1.2"

```

### Usage

Use LangGraph workflows within an `AgentSession` by wrapping them with the `LLMAdapter`. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from langgraph.graph import StateGraph
from livekit.agents import AgentSession, Agent
from livekit.plugins import langchain

# Define your LangGraph workflow
def create_workflow():
    workflow = StateGraph(...)
    # Add your nodes and edges
    return workflow.compile()

# Use the workflow as an LLM
session = AgentSession(
    llm=langchain.LLMAdapter(
        graph=create_workflow()
    ),
    # ... stt, tts, vad, turn_detection, etc.
)

```

The `LLMAdapter` automatically converts the LiveKit chat context to [LangChain messages](https://python.langchain.com/docs/concepts/messages/#langchain-messages). The mapping is as follows:

- `system` and `developer` messages to `SystemMessage`
- `user` messages to `HumanMessage`
- `assistant` messages to `AIMessage`

### Parameters

This section describes the available parameters for the `LLMAdapter`. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/langchain/index.html.md#livekit.plugins.langchain.LLMAdapter) for a complete list of all available parameters.

- **`graph`** _(PregelProtocol)_: The LangGraph workflow to use as an LLM. Must be a locally compiled graph. To learn more, see  [Graph Definitions](https://langchain-ai.github.io/langgraph/reference/graphs/).

- **`config`** _(RunnableConfig | None)_ (optional) - Default: `None`: Configuration options for the LangGraph workflow execution. This can include runtime configuration, callbacks, and other LangGraph-specific options. To learn more, see [RunnableConfig](https://python.langchain.com/docs/concepts/runnables/#runnableconfig).

## Additional resources

The following resources provide more information about using LangChain with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-langchain/)**: The `livekit-plugins-langchain` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/langchain/index.html.md#livekit.plugins.langchain.LLMAdapter)**: Reference for the LangChain LLM adapter.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-langchain)**: View the source or contribute to the LiveKit LangChain plugin.

- **[LangChain docs](https://python.langchain.com/docs/)**: LangChain documentation and tutorials.

- **[LangGraph docs](https://python.langchain.com/docs/langgraph)**: LangGraph documentation for building stateful workflows.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and LangChain.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/plugins/langchain.md](https://docs.livekit.io/agents/models/llm/plugins/langchain.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).