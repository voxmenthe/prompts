basic_cot = """Approach this problem step by step, using the following examples as a general indicator of how to solve problems, but not as a strict rule.
Examples:
{examples}
"""

basic_zeroshot_cot_prompt = """Approach this step by step, thinking about what steps to take first before directly tackling the problem.
Then, proceed with solving the intermediate steps first, keeping your work organized and checking your work as you go.
Finally, proceed with solving the main problem or final steps.
"""

rephrase_and_repeat = """Before proceeding, repeat the task, or at least its key components, and then rephrase and expand on the problem in your own words, thinking out loud about all the steps you will need to take to complete the task.
Then, proceed with the task.
"""

fan_out_thinking = """Brainstorm about the key components of the problem and the steps you will need to take to complete the task.
Then, expand on each step and think about all the different ways you can solve each step.
Finally, select the best approach for each step and proceed with the task.
"""

least_to_most = """Break down the problem into smaller sub-problems, and then rank them in order of increasing difficulty, from the most straightforward, to the most subtle, complex and difficult. Start by solving all of the simple and medium difficulty sub-problems, then, based on the results you have so far, and everything you know, make a plan to solve the subtler more complex problems, and then work on those in order of increasing difficulty."""
