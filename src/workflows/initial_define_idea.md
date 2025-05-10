I start with a simpler model (4o), not to code, just to brainstorm. I talk through the idea, refine what I actually want, like, “I wanna recreate this effect...” or “I know dithering works like this...” No strict structure, just discuss about concept.

Once things make sense, I ask it to format everything into a clear, structured task (like for a dev). Then I take that, open a fresh chat with a stronger model (o3), and drop it in


=================================

the fastest way to clone an app

1. screenshot all the pages of an app
2. paste them into copycoder
3. get the prompts to build your app in cursor

=================================

# "iterating architect + fresh implementer"

- write plan with o3
- try implementing plan with weaker model. sometimes, we discover new stuff and realize the plan was wrong!
- revert changes, go back to o3 and explain the issue, revise the plan
- start a new implementation from the top using the new plan

feels nice because:

- the o3 chat builds up lots of useful context, but it's just iterations on the high level plan, no implementation noise
- each implementation attempt starts fresh, w/ just the starting codebase + a good plan

def overkill for simple stuff, but subjectively seems useful to me for harder changes