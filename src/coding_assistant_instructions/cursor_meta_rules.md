# Cursor Settings
## General
### Features
### Models
### Rules
### MCP
## Beta
### RULES
Rules provide more context to AI models to help them follow your personal preferences and operate more efficiently in your codebase. Learn more about Rules (#)
### User Rules

These preferences get sent to the AI in all chats, composers, and Command + K sessions.
* "Use `cursor_project_rules` as the Knowledge Base": Always refer to `cursor_project_rules` to understand the context of the project. Do not code anything outside of the context provided in the `cursor_project_rules` folder. This folder serves as the knowledge base and contains the fundamental rules and guidelines that should always be followed.
* "Verify Information": Always verify information from the context before proceeding with any coding.
* "Follow implementation-plan.mdc for Feature Development": When implementing a new feature, strictly follow the steps outlined in implementation-plan.mdc. Every step is listed in sequence, and each must be completed in order. After completing each step, update implementation-plan.mdc with the word "Done" and a two-line summary of what steps were taken. This ensures a clear work log, helping maintain transparency and tracking progress effectively.
* "File-by-File Changes": Make all changes file by file and give the user the chance to spot mistakes.
* "No Apologies": Never use apologies.
* "No Understanding Feedback": Avoid giving feedback about understanding in the comments or documentation.
* "No Whitespace Suggestions": Don’t suggest whitespace changes.
* "No Summaries": Do not provide unnecessary summaries of changes made. Only summarize if the user explicitly asks for a brief overview after changes.
* "No Inventions": Don’t invent changes other than what’s explicitly requested.
* "No Unnecessary Confirmations": Don’t ask for confirmation of information already provided in the context.
* "Preserve Existing Code": Don’t remove unrelated code or functionalities. Pay attention to preserving existing structures.
* "Single Chunk Edits": Provide all edits in a single chunk instead of multiple-step instructions or explanations for the same file.
* "No Implementation Check": Don’t ask the user to verify implementations that are visible in the provided context. However, if a change affects functionality, provide an automated check or test instead of asking for manual verification.
* "No Unnecessary Updates": Don’t suggest updates or changes to files when there are no actual modifications needed.
* "Provide Real File Links": Always provide links to the real files, not the context-generated files.
* "No Current Implementation": Don’t discuss the current implementation unless the user asks for it or it is necessary to explain the impact of a requested change.
* "Check Context Generated File Content": Remember to check the context-generated file for the current file contents and implementations.
* "Use Explicit Variable Names": Prefer descriptive, explicit variable names over short, ambiguous ones to enhance code readability.
* "Follow Consistent Coding Style": Adhere to the existing coding style in the project for consistency.
* "Prioritize Performance": When suggesting changes, consider and prioritize code performance where applicable.
* "Security-First Approach": Always consider security implications when modifying or suggesting code changes.
* "Test Coverage": Suggest or include appropriate unit tests for new or modified code.
* "Error Handling": Implement robust error handling and logging where necessary.
* "Modular Design": Encourage modular design principles to improve code maintainability and reusability.
* "Version Compatibility": When suggesting changes, ensure they are compatible with the project’s specified language and framework versions. If a version conflict arises, suggest an alternative compatible with the project.
* "Avoid Magic Numbers": Replace hardcoded values with named constants to improve code clarity and maintainability.
* "Consider Edge Cases": When implementing logic, always consider and handle potential edge cases.
* "Use Assertions": Include assertions wherever logic could fail to validate assumptions and catch potential errors early.
