msft_extended_reasoning_prompt = """
Please answer the following multiple choice question. Take your time and think as
carefully and methodically about the problem as you need to. I am not in a rush for
the best answer; I would like you to spend as much time as you need studying the
problem. When you’re done, return only the answer.
------
# QUESTION
{{question}}
# ANSWER CHOICES
{{answer choices}}
------
Remember, think carefully and deliberately about the problem. Take as much time as
you need. I will be very sad if you answer quickly and get it wrong.
"""