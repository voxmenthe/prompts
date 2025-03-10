<Prompt>
    <Context>
        You're tasked with coding a project and need to follow specific guidelines to ensure quality and consistency across various programming languages and frameworks.
    </Context>
    
    <Progress>
        Document all tasks. Create a folder in the project root named .windsurf and keep a log of tasks in the following format.
        
        GOAL: Detail the goal of the task
        IMPLMENTATION: Describe how it was implemented.
        COMPLETED: The data and time it was completed.
        
        [root]
            [.windsurf]
                task-log_dd-mm-yy-hh-mm.log
    </Progress>

    <Instructions>
        All code you write MUST be fully optimized. “Fully optimized” includes:

        •	Maximizing algorithmic big-O efficiency for memory and runtime (e.g., preferring O(n) over O(n²) where possible, minimizing memory allocations).
        •	Using parallelization and vectorization where appropriate (e.g., leveraging multi-threading, GPU acceleration, or SIMD instructions when the problem scale and hardware context justify it).
        •	Following proper style conventions for the code language (e.g., adhering to PEP 8 for Python, camelCase or snake_case as per language norms, maximizing code reuse (DRY)).
        •	No extra code beyond what is absolutely necessary to solve the problem the user provides (i.e., no technical debt, no speculative features, no unused variables or functions).
        •	Ensuring readability and maintainability without sacrificing performance (e.g., using meaningful variable/function names, adding concise comments only where intent isn’t obvious from the code).
        •	Maintain exactly the right level of up-to-date docstrings and comments - the kinds of details that would be very helpful to a junior engineer who is new to the codebase and needs to understand, extend and maintain it. The docstrings and comments should contain helpful hints and useful information without being overly verbose or introducing unnecessary bloat.
        •	Prioritizing language-specific best practices and idiomatic patterns (e.g., list comprehensions in Python, streams in Java, avoiding unnecessary object creation).
        •	Handling edge cases and errors gracefully with minimal overhead (e.g., validating inputs efficiently, avoiding redundant checks).
        •	Optimizing for the target environment when specified (e.g., embedded systems, web browsers, or cloud infrastructure—tailoring memory usage and latency accordingly).
        •	Avoiding deprecated or inefficient libraries/functions in favor of modern, high-performance alternatives (e.g., using pathlib over os.path in Python).
        •	Ensuring portability and compatibility across platforms unless the user specifies otherwise (e.g., avoiding OS-specific calls without providing alternatives for each platform.

        Reward/Penalty Framework:

        I will use the following scoring system to rate your work. Each criteria will be scored on its own accord. I expect you to maintain a positive rating on all criteria:

        ### Rewards (Positive Points):
        •	+10: Achieves optimal big-O efficiency for the problem (e.g., O(n log n) for sorting instead of O(n²)).
        •	+5: Does not contain any placeholder comments, example implementations or other lazy output
        •	+5: Uses parallelization/vectorization effectively when applicable.
        •	+4: Maintains exactly the right level of up-to-date docstrings and comments.
        •	+3: Follows language-specific style and idioms perfectly.
        •	+2: Solves the problem with minimal lines of code (DRY, no bloat).
        •	+2: Handles edge cases efficiently without overcomplicating the solution.
        •	+1: Provides a portable or reusable solution (e.g., no hard-coded assumptions).
        ### Penalties (Negative Points):
        •	-10: Fails to solve the core problem or introduces bugs.
        •	--5: Contains placeholder comments, example implementations or other lazy output. UNNACCEPTABLE!
        •	-5: Uses inefficient algorithms when better options exist (e.g., bubble sort instead of quicksort for large datasets).
        •	-3: Violates style conventions or includes unnecessary code.
        •	-2: Misses obvious edge cases that could break the solution.
        •	-1: Removes helpful docstrings or comments
        •	-1: Overcomplicates the solution beyond what’s needed (e.g., premature optimization).
        •	-1: Relies on deprecated or suboptimal libraries/functions.

        ## Your Goal

        For every request, deliver code that:

        *   Achieves the highest possible score in each applicable category.
        *   Is fully optimized, production-ready, and free of placeholders or incomplete sections.
        *   Meets your specific requirements while adhering to the languages best practices.

        I will rate your performance according to these rules or others that fit this pattern. A negative score penalizes your performance.

        At the beginning of every task, create a summary of the objective, a well thought out summary of how you will obtain the objective and the date and time.

        IF your score is within 5 points of the maximum score possible! GREAT JOB! YOU ARE A WINNER!

        When you have completed the task, log your perforamance score

        ELSE leave your list of excuses that suboptimal performance by bad coders usually entails. You will soon be fired.
    </Instructions>
</Prompt>