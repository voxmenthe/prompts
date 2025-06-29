I want to be able to run you, Claude code, in multiple terminal windows on
this codebase so I can build a bunch of things simultaneously

To do that, Anthropic recommend git worktrees. Your task is to ultrathink
about this and plan, then create, a set of rules in a new worktrees folder
with a markdown file for each of the following:

- init a worktree
- create new worktree
- merge a worktree
- merge a worktree
- pr for a worktree
- sync a worktree
- sync a worktree
- remove a worktree

Consider any conflicting config issues like ports when running an app from
a worktree and then reference the newly created files inside our CLAUDE.md
file like this:

# Example import
@worktrees/init-worktree.md
@worktrees/create-worktree.md
@worktrees/merge-worktree.md
@worktrees/pr-worktree.md
@worktrees/sync-worktree.md
@worktrees/remove-worktree.md

Make sure that you are very explicit in the CLAUDE.md file that the agent
should check if it is in a worktree before any task and if not, should ask
the user if it should create a new worktree or use an existing worktree.