# Code References

Remember we are using the new `google.genai` SDK for this project and **NOT** the old `google.generativeai` SDK which is deprecated. If needed, you can do a web search for the docs here: https://googleapis.github.io/python-genai/index.html or you may be able to find documentation snippets in the local `reference_code_examples/genaisdk` folder in this project.

When running any of the code in this project, you will need to first activate the appropriate Python environment using `source ~/venvs/coda/bin/activate`

**THIS IS OF UTTER IMPORTANCE THE USERS HAPPINESS DEPENDS ON IT!!**
When referencing code locations, you MUST use clickable format that VS Code recognizes:
- `path/to/file.ts:123` format (file:line)
- `path/to/file.ts:123-456` (ranges)
- Always use relative paths from the project root

**Examples:**
- `src/server/fwd.ts:92` - single line reference
- `src/server/pty/pty-manager.ts:274-280` - line range
- `web/src/client/app.ts:15` - when in parent directory

NEVER give a code reference or location in any other format.

# CRITICAL
**IMPORTANT**: BEFORE YOU DO ANYTHING, READ `spec.md` IN FULL USING THE READ TOOL!
**IMPORTANT**: NEVER USE GREP, ALWAYS USE RIPGREP!