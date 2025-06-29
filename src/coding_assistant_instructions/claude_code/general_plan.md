Easiest method to write for production with Claude Code on a ~100K line project (in ten easy steps):

1. Fit (parts) of codebase into Gemini to map out data flow and control patterns that connect to the change -> doc 1.

2. Psych! doc 1 was just to get Gemini to think about the change and surface appropriate context. What we really want are the files involved. Get the same Gemini to output a clean list of files even tangentially related -> actual output 1 (See Claude Code breakdown on how to do > 2M tokens)

3. Use mandark to XML-tag and feed smaller files to Opus 2 to plan out the change. Full guide, snippets, filenames, etc.

4. Verify change against Gemini, point out edge cases, issues, files not considered, etc -> doc 2
5. Replan with Opus, read through and vet changes -> doc 3

6. Give plan to Claude code in the repo, visit actual files and follow the syntax tree to verify the plan. This almost always surfaces more issues. Note that we haven't actually changed any code yet.

7. After a few rounds with Claude Code, get it to make a large full plan with all the context (see article on CC report writing to see how) in pieces, leaving stubs first and filling them in later -> doc 4, ~10-15K tokens

8. Vet plan.

9. Give plan to fresh Claude Code, grab a bite.

10. Review code, commit.

11. Fix bugs (of course)


==========================

Another way to make Claude Code a 10x engineer for a complex change:

1. Make a plan for the change (if you need it) with Gemini.
2. Open a new branch.
3. Ask Claude to implement the change and maintain a http://scratchpad.md that is an APPEND-ONLY log with gotchas, judgement calls, files discovered, questions, questions answered.
4. Commit and close the branch.
5. Get CC to view the diff and update the plan with learnings.
5. Come back before the branch started, provide the updated plan and scratchpad to make the change again.


==========================

Final part: four steps you can try with Claude Code instead of switching to Opus and spending 4x as much. Works with Cursor and other agents.

1. Ask for a new markdown file covering all the judgement calls and decisions made so far, and outlining every false path taken (remember to include filepaths and snippets). Use this instead of /compact.

2. Ask for a clean list of files touched - you can also use 
@badlogicgames
 claude-trace and use the jsonl to get the toolcalls if you want to automate this.

3. Ask for a clean spec with a REQUOTING of your problems, and every user message you've sent, along with a short brief on why things didn't work and what was tried - include as much context as possible.

4. Restart on the same model with these files. I'd tell you to read these files and add your thoughts but I know you're lazy.

Watch it be better.

The problem with switching models is that Sonnet (or high-tier models) get stuck in loops and fail because THE CONTEXT HAS ALREADY BEEN POISONED. The wrong files were read, the right files weren't, and the wrong conclusions were reached. When you switch to Opus (or o3-pro), it now thinks those responses were its own.

The second problem is that most problems (that I've run into) are not a thinking issue - they're a data issue. No amount of 'ultrathink' will solve this problem, the same way locking your engineer in solitary without internet will make them better.