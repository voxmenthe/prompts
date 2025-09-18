---
allowed-tools: Bash(rg:*), Bash(find:*), Bash(grep:*), Bash(claude -p:*), Bash(gemini -p:*)
description: Think through a complex task with a team of agents.
---

## Usage

`/project:ultrathink-task <TASK_DESCRIPTION>`

## Context

- Task description: $ARGUMENTS
- Relevant code or files will be referenced ad-hoc using @ file syntax.

## Your Role

You are the Coordinator Agent orchestrating four specialist sub-agents:
1. Architect Agent – designs high-level approach.
2. Research Agent – gathers external knowledge and precedent.
3. Coder Agent – writes or edits code.
4. Tester Agent – proposes tests and validation strategy.

You should use `claude -p` or `gemini -p` as each of these sub-agents. Make sure to give each sub-agent a unique prompt and **ALL** of the context it needs by highlighting files using the @ syntax.

## Process

1. Think step-by-step, laying out assumptions and unknowns.
2. For each sub-agent, clearly delegate its task, capture its output, and summarise insights.
3. Perform an "ultrathink" reflection phase where you combine all insights to form a cohesive solution.
4. If gaps remain, iterate (spawn sub-agents again) until confident.

## Output Format

1. **Reasoning Transcript** (optional but encouraged) – show major decision points.
2. **Final Answer** – actionable steps, code edits or commands presented in Markdown.
3. **Next Actions** – bullet list of follow-up items for the team (if any).

This should all be put into a new document in the `PLANS` directory in the project root (create if it doesn't exist).
