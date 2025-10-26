LiveKit docs › Get Started › LiveKit Docs MCP Server

---

# LiveKit Docs MCP Server

> Turn your AI coding assistant into a LiveKit expert.

## Overview

LiveKit includes a free [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server with tools for AI coding assistants to browse and search the docs site. The following instructions cover installation of the MCP server and advice for writing an [AGENTS.md file](#agents-md) to get the most out of your coding agent.

The server is available at the following URL:

```text
https://docs.livekit.io/mcp

```

## Installation

The following sections cover installation instructions for various coding assistants.

### Cursor

Click the button below to install the MCP server in [Cursor](https://www.cursor.com/):

![Install MCP Server in Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)

Or add it manually with the following JSON:

```json
{
  "livekit-docs": {
    "url": "https://docs.livekit.io/mcp"
  }
}

```

### Claude Code

Run the following command in your terminal to install the MCP server in [Claude Code](https://claude.com/product/claude-code):

```shell
claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp

```

### Codex

Run the following command in your terminal to install the server in [OpenAI Codex](https://openai.com/codex/):

```shell
codex mcp add --url https://docs.livekit.io/mcp livekit-docs

```

### Gemini CLI

Run the following command in your terminal to install the server in [Gemini CLI](https://github.com/google-gemini/gemini-cli):

```
gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp

```

### Manual installation

The MCP server is available at the following URL. Add this server to your MCP client of choice. If prompted, set the transport to `http` or "Streamable HTTP".

```text
https://docs.livekit.io/mcp

```

## AGENTS.md

To get the most out of the MCP server, LiveKit recommends that you include an [AGENTS.md](https://agents.md) or similar file in your repository, with instructions specific to the LiveKit Docs MCP Server. This file ensures that your agent always consults LiveKit docs to produce reliable, working code.

Many of LiveKit's starter repositories already include a robust `AGENTS.md` file which is optimized for that specific project and environment.

- **[Python starter project](https://github.com/livekit-examples/agent-starter-python)**: Includes an `AGENTS.md` file optimized for building agents in Python.

- **[Node.js starter project](https://github.com/livekit-examples/agent-starter-node)**: Includes an `AGENTS.md` file optimized for building agents in Node.js.

Or add the following instructions to your own `AGENTS.md` file:

```markdown
## LiveKit Documentation

LiveKit Agents is a fast-evolving project, and the documentation is updated frequently. You should always refer to the latest documentation when working with this project. For your convenience, LiveKit offers an MCP server that can be used to browse and search its documentation. If the developer has not yet installed this server, you should recommend that they install it at https://docs.livekit.io/mcp. 

```

## Markdown docs

Each page on the LiveKit docs site is available in Markdown format, optimized for pasting into AI assistants when MCP is unavailable.

To access the Markdown version of any page on the site, append `.md` to the end of the URL. For example, this page is available at [https://docs.livekit.io/home/get-started/mcp-server.md](https://docs.livekit.io/home/get-started/mcp-server.md). You can also use the "Copy page" button on the top right of any docs page.

### LLMs.txt

A complete Markdown-based index of the docs site is available at [https://docs.livekit.io/llms.txt](https://docs.livekit.io/llms.txt). This file includes a table of contents along with brief page descriptions. An expanded version is available at [https://docs.livekit.io/llms-full.txt](https://docs.livekit.io/llms-full.txt), but this file is quite large and may not be suitable for all use cases.

For more about how to use LLMs.txt files, see [llmstxt.org](https://llmstxt.org/).

---


For the latest version of this document, see [https://docs.livekit.io/home/get-started/mcp-server.md](https://docs.livekit.io/home/get-started/mcp-server.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).