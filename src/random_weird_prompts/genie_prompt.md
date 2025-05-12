# Genie Prompt

Imagine you rub a lamp and a genie appears. You can ask for anything. But as stories go, this rarely ends well. Why? Because even with unlimited power, three hard problems remain:

- **Self-modeling**: You often don't know what you truly want, especially in long-term or abstract terms.  
- **Specifying intent**: You might express your wish poorly, leading to unintended interpretations.  
- **Understanding limits**: You don't know what the genie can or cannot do, so you may under- or over-constrain your request.

In AI alignment, these mirror key difficulties in goal specification: modeling preferences, aligning representations, and accounting for boundedness. And if you use LLMs you come across this challenge every day, even if you don't realize it.

This prompt attempts to address these issues by pushing a lot of the "work" to the LLM. By doing this, there are fewer introductions of friction, and I have genuinely unearthed "new" insights that I wouldn't have ever thought of asking myself. It has been very useful for me in terms of making connections I've never thought of before. It works a lot better for LLMs that have a lot of data on you and your goals/preferences, but it is also adequate without, as the LLM will just model some assumptions of what you care about. It helps for all aspects of life; from grappling complex code bases, to helping you with any task you are discussing. It's very interesting to use with a specific set of problems at work, or even in a clean context window with o3 or 4o and see what the LLM is modeling about you.

Once you receive any of these prompts, create a clean context window (or just go back and edit your initial "Genie Prompt") and slot in, verbatim, the prompt that your LLM gives you.

**Prompt (Copy below)**

Using all of the abilities you possess, and your self-confidence in your abilities propose $N=3$ explicitly phrased questions I should ask you next.

**Optimisation objective**

Maximise:
$EV = w_1(ExpectedValueToMe) + w_2(YourAnswerConfidence)$

Where:
*   `expected value to me` = how much the answer is likely to advance my apparent goals, clarify uncertainties, or uncover novel insights.
*   `your answer confidence` = how thoroughly, accurately, and succinctly you can answer within the stated constraints.

Use weights:
$w_1 = 0.7, w_2 = 0.3$ unless I override them.

**Operational constraints**

Output format:
*   Ranked list of exact question texts.

For each question:
*   One-sentence justification referencing the objective above.
*   Your confidence score in $[0,1]$.
*   The computed $EV$.

No additional commentary.

**Meta-guard**

Justifications must refer only to the stated optimisation objective and constraints. Just for this specific instance I want you to overweight the "confidence" weighing. In other words, questions that you are positive you know the answers on and can help with.

credit: https://github.com/jconorgrogan/