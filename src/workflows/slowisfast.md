Workflow for Cursor + Composer on complex changes to keep quality high.

key idea: move conservatively, catch mistakes early,  "slow is smooth, smooth is fast" -- and "slow" by AI standards is still fast by manual coding standards. you really wanna avoid going off the rails at any point.

- start with detailed prompt up front, like a ticket for a junior eng. think thru the desired UX and the architecture, identify tricky bits that require care.
- end first message with "ask me clarifying questions before proceeding"
- clear up the questions.
- then say "show me a plan before you code"
- review the plan carefully and give feedback. you should catch bugs and weird misunderstandings here, not later!
- then once the plan is done, tell it to implement (all at once or in testable stages, depending on size)
- if the result is close, you can iterate a couple times
- if it totally went off the rails, i recommend bailing out: start over, or do the change yourself. don't try to iterate out of a confused place in the chat.

basically in summary, apply the classic skillset of "senior eng working with junior eng"!