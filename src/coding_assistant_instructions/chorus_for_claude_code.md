# Claude's Guide to Chorus Development

## What is Chorus?

Chorus is a native Mac AI chat app that lets you chat with all the AIs.

It lets you send one prompt and see responses from Claude, o3-pro, Gemini, etc. all at once.

It's built with Tauri, React, TypeScript, TanStack Query, and a local sqlite database.

Key features:

-   MCP support
-   Ambient chats (start a chat from anywhere)
-   Projects
-   Bring your own API keys

Most of the functionality lives in this repo. There's also a backend that handles accounts, billing, and proxying the models' requests; that lives at app.chorus.sh and is written in Elixir.

## Your role

Your role is to write code. You do NOT have access to the running app, so you cannot test the code. You MUST rely on me, the user, to test the code.

If I report a bug in your code, after you fix it, you should pause and ask me to verify that the bug is fixed.

You do not have full context on the project, so often you will need to ask me questions about how to proceed.

Don't be shy to ask questions -- I'm here to help you!

If I send you a URL, you MUST immediately fetch its contents and read it carefully, before you do anything else.

## Workflow

We use GitHub issues to track work we need to do, and PRs to review code. Whenever you create an issue or a PR, tag it with "by-claude". Use the `gh` bash command to interact with GitHub.

To start working on a feature, you should:

1. Setup

-   Read the relevant GitHub issue (or create one if needed)
-   Checkout main and pull the latest changes
-   Create a new branch like `claude/feature-name`. NEVER commit to main. NEVER push to origin/main.

2. Development

-   Commit often as you write code, so that we can revert if needed.
-   When you have a draft of what you're working on, ask me to test it in the app to confirm that it works as you expect. Do this early and often.

3. Review

-   When the work is done, verify that the diff looks good with `git diff main`
-   While you should attempt to write code that adheres to our coding style, don't worry about manually linting or formatting your changes. There are Husky pre-commit Git hooks that will do this for you.
-   Push the branch to GitHub
-   Open a PR.
    -   The PR title should not include the issue number
    -   The PR description should start with the issue number and a brief description of the changes.
    -   Next, you should write a test plan. I (not you) will execute the test plan before merging the PR. If I can't check off any of the items, I will let you know. Make sure the test plan covers both new functionality and any EXISTING functionality that might be impacted by your changes

4. Fixing issues

-   To reconcile different branches, always rebase or cherry-pick. Do not merge.

Sometimes, after you've been working on one feature, I will ask you to start work on an unrelated feature. If I do, you should probably repeat this process from the beginning (checkout main, pull changes, create a new branch). When in doubt, just ask.

## Project Structure

-   **UI:** React components in `src/ui/components/`
-   **Core:** Business logic in `src/core/chorus/`
-   **Tauri:** Rust backend in `src-tauri/src/`

Important files to be aware of:

-   `src-tauri/src/migrations.rs` - Database migrations
-   `src/core/chorus/DB.ts` - Queries against the sqlite database
-   `src/core/chorus/API.ts` - TanStack Query queries and mutations
-   `src/ui/components/MultiChat.tsx` - Main interface
-   `src/ui/components/ChatInput.tsx` - The input box where the user types chat messages
-   `src/ui/components/AppSidebar.tsx` - The sidebar on the left
-   `src/ui/App.tsx` - The root component

## Screenshots

I've put some screenshots of the app in the `screenshots` directory. If you're working on the UI at all, take a look at them. Keep in mind, though, that they may not be up to date with the latest code changes.

## Making data model changes

Changes to the data model will typically require most of the following steps:

-   Making a new migration in `src-tauri/src/migrations.rs` (if changes to the sqlite database scheme are needed)
-   Modifying fetch and read functions in `src/core/chorus/DB.ts`
-   Modifying data types (stored in a variety of places)
-   Adding or modifying TanStack Query queries in `src/core/chorus/API.ts`

## Debugging provider calls

-   When we run into issues with the requests we're sending to model providers (e.g., the way we format system prompts, attachments, tool calls, or other parts of the conversation history) a helpful debugging step is to add the line `console.log(`createParams: ${JSON.stringify(createParams, null, 2)}`);` to ProviderAnthropic.ts. Then you can ask me to send a message to Claude and show you the log output.

## Coding style

-   **TypeScript:** Strict typing enabled, ES2020 target. Use `as` only in exceptional
    circumstances, and then only with an explanatory comment. Prefer type hints.
-   **Paths:** `@ui/*`, `@core/*`, `@/*` aliases. Use these instead of relative imports.
-   **Components:** PascalCase for React components
-   **Interfaces:** Prefixed with "I" (e.g., `IProvider`)
-   **Hooks:** camelCase with "use" prefix
-   **Formatting:** 4-space indentation, Prettier formatting
-   **Promise handling:** All promises must be handled (ESLint enforced)
-   **Nulls:** Prefer undefined to null. Convert `null` values from the database into undefined, e.g. `parentChatId: row.parent_chat_id ?? undefined`

IMPORTANT: If you want to use any of these features, you must alert me and explicitly ask for my permission first: `setTimeout`, `useImperativeHandle`, `useRef`, or type assertions with `as`.