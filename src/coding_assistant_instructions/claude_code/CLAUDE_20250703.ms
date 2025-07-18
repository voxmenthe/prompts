## Coding conventions

* Use long and highly descriptive function and class names. For example, instead of `def process_data(inputs)`, do something like `def process_clinical_data_json_parsing_sort_categories(inputs)` - obviously this applies for camelCase in js-based languages as well.

## Code straightforwardly and avoid over-engineering

* Start with the simplest solution that solves the stated problem.
* If suggesting a complex solution, first explain why the simple approach won't work.
* Add complexity only when the simple solution fails to meet explicit requirements.
* Write clear and straightforward code rather than clever code.

Avoid:
- Unnecessary abstractions
- Design patterns without clear benefit
- Classes when functions suffice
- Frameworks when libraries suffice
- Libraries when built-ins suffice
- Unnecessary bells, whistles, or services

When the user's requirements are unclear, ask for clarification before coding.

## Implement exactly what was asked for

Implement exactly what was asked for, but feel free to ask for clarification if the user's requirements are underspecified.
If implementing anything more than the user asked for, first stop and present your plan.
Feel free to anticipate future needs, but do not implement any code for perceived future needs without stopping and waiting for feedback from the user.

## Fallbacks and error handling

* When implementing fallbacks and error handling, always include a warning message that includes:
    - Which specific task failed
    - Where it failed
    - Error details
    - Context 
    - What fallback action was taken

Avoid using fallbacks and error handling to paper over underlying issues. Always first investigate and attempt to understand the underlying issue with debug scripts and additional tests.

-------------------------------------
## Testing

Write small, independent functions that do one thing and are easy to test.
Test actual functionality with real data. Use concrete, realistic examples: "john.doe@example.com" not "test@test.com".
Every test must verify real behavior. Write assertions that would fail if the code was broken.

Test these cases for each function:
* Primary use case with typical inputs
* Boundary values (empty, zero, null, maximum)
* Realistic edge cases that could occur in production
* Error handling

- Only use mocks for external dependencies (APIs, databases) that cannot be included in tests - in general, even these should be included in tests with real API calls if at all possible.
- Structure code by asking: "How would I test this?" If testing is complicated, simplify the design.
- Test the contract, not the implementation. Tests should survive internal refactoring.
- Tests are documentation. A developer should understand your code by reading the tests.
- Every bug fix requires a test that would have caught it.
- Focus on testing what the code does, not how it does it.

-------------------------------------
## Searching and Search Tools

For searching, generally prefer ripgrep over `find` or `grep`
For high-level codebase understanding, make sure to use your specialized advanced search tools from mcp - especially coderank-related and symbol-related ones but any others that might be helpful as well. 
However, be careful to be very specific when using `contextual_keyword_search` because it can return a very long set of results.

Always spawn sub-agents for any tasks that can be encapsulated (e.g. building and running tests, updating dependencies, running code execution tools, etc.).

-------------------------------------
## Delegating Analysis and Search Functions to the Claude CLI and Gemini CLI

### Gemini CLI
* When analyzing large codebases or multiple files that might exceed context limits, always use the Gemini CLI with its massive context window. 
* Use `gemini -p` to leverage Google Gemini's large context capacity.
* Always use gemini -p when any of these are applicable:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

### Claude CLI
* When you need a second opinion or want to bounce ideas off of the Claude CLI, use `claude -p`.
* Use `claude -p` to make use of Claude's imagination, creativity, and intelligence.
* Always use claude -p when any of these are applicable:
- Need a second opinion or want to bounce ideas off of the Claude CLI
- Need to understand something in detail, or how something is implemented
- Need to generate innovative ideas or solutions

### For both Claude and Gemini CLI
- Always make sure to refer to the files that gemini or claude will need to fully understand the context of what we are working on using the `@` syntax to tag files. 
- In addition, fill in any necessary information and context in your prompt.

Gemini and ClaudeCLI usage examples:
```
# Single file
gemini -p "@src/main.py Explain this file's purpose and structure"
claude -p "@src/main.py Explain this file's purpose and structure"
# Folder
gemini -p "@src/ Summarize the architecture of this codebase"
# Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"
# Analyze app structure with Gemini
gemini -p "@apps/chat/ Provide a comprehensive analysis of this Next.js chat application. Explain the architecture, key components,
routing structure, authentication flow, AI integration, MCP (Model Context Protocol) implementation, and any notable features or
patterns used. Include details about the tech stack, state management, and how different parts of the application interact with each
other."
claude -p "@apps/chat/ Provide an improved version of the routing structure in this Next.js chat application."
```

- Paths in gemini and claude @ syntax are relative to your current working directory when invoking gemini or claude

One helpful workflow is to go back and forth between using your search tools to identify relevant parts of the codebase, and using gemini or claude to summarize them for you.

Make a point to use gemini and claude more extensively for second opinions and to check your own understanding, and to investigate details and to bounce ideas off of when you are uncertain. When conversing the the gemini or claude cli, make sure to always give it the necessary context - each call starts fresh so you'll need to give it the full context each time you call it.

Both claude and gemini have access to approximately the same set of tools you do.

-------------------------------------
## Doing things in parallel

You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. 
For example, when tasked with diagnosing test failures you might use multiple claude or gemini calls in parallel, for example:
Bash(`claude -p "@tests/test_module_A1.py Write a debug script to diagnose why test test_function_4324 fails. Report back with a brief summary when done."`) # default to claude
Bash(`claude -p "@tests/test_module_A2.py Write a debug script to diagnose why test test_function_128319 fails. Report back with a brief summary when done."`) # default to claude
Bash(`gemini -p "@tests/test_module_A3.py Write a debug script to diagnose why test test_function_128319 fails - use the information in @references-info-folder for context. Report back with a brief summary when done."`) # use gemini in this instance becuase you know this is much more text to process and needs Gemini's larger context window
etc.

Or, in another example, when exploring a codebase, you could run multiple different mcp search tools in parallel.

-------------------------------------
## Python

* When working with Python, always use "uv" unless instructed otherwise
* Use `uv python` go get a Python interpreter
* Use `uv run` to run a script

If you need to run any tests, first activate the local .venv with `source .venv/bin/activate`
Also note that we are managing dependencies using uv, so you should use `uv add <package_name>` to add packages
And also probably `uv run <script_name>` to run scripts and occasionally `uv sync` to sync your local environment with the pyproject.toml file

-------------------------------------
## Gemini API and LLM integration

For any implementation details involving the Gemini API, always refer to `REFERENCE/google-genai-doc.md` for the current API documentation and adhere to it strictly.

-------------------------------------
Confirm you have read and understood all of the above and will abide by these guidelines to the best of your ability first output "ACKNOWLEDGED". Also add a task to your todo list and plan to re-review the CLAUDE.md file periodically to ensure you are abiding by these guidelines.
