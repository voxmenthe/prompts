## Standard Chain-of-Thought Prompting

This is the original CoT technique introduced by Wei et al. in 2022. It involves providing the LLM with examples that include step-by-step reasoning before the final answer[1]. This encourages the model to generate similar reasoning steps for new problems, often leading to more accurate results.

## Zero-Shot Chain-of-Thought

Introduced shortly after the original CoT paper, this technique doesn't require exemplars. Instead, it simply appends phrases like "Let's approach this step by step:" to the end of prompts[1]. This can activate reasoning capabilities in LLMs without needing specific examples.

## Self-Consistency

This method generates multiple reasoning chains for a single problem and then selects the most consistent answer[2]. It helps mitigate errors in individual reasoning chains by leveraging the consensus across multiple attempts.

## Tree of Thoughts

An extension of CoT that explores multiple reasoning paths simultaneously, creating a tree-like structure of thoughts[3]. This allows for backtracking and considering alternative reasoning routes, potentially leading to better solutions for complex problems.

## Algorithm of Thoughts

This approach combines CoT with algorithmic thinking, encouraging LLMs to break down problems into sub-problems and solve them systematically[4]. It's particularly useful for tasks that require structured problem-solving approaches.

## Active-Prompt

This technique uses uncertainty-based active learning to select the most informative examples for CoT prompting[1]. It aims to create more effective prompt sets by focusing on challenging yet solvable problems.

## Least-to-Most Prompting

This method breaks down complex problems into simpler sub-problems, solving them in order of increasing difficulty[5]. It's especially useful for tasks that can be naturally decomposed into simpler components.

## Iterative Refinement

This approach involves generating an initial chain of thought, then iteratively refining it to improve accuracy and coherence[6]. It can help correct errors and enhance the quality of the reasoning process.

## Self-Play

This method involves training an LLM to play a game by itself, using the output from the LLM as the input for the next round of play[7]. It can help the LLM improve its performance by allowing it to learn from its own reasoning processes.

## Chain of Thought with Output-Fixing

This approach combines CoT with output-fixing, a technique that corrects errors in the LLM's output[8]. It can help the LLM produce more accurate results by providing explicit feedback on its output.