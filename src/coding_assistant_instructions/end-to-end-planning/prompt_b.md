Here are four comprehensive prompts designed for an AI coding assistant, drawing upon principles from the attached research papers to guide it through a structured process of codebase research, planning, and implementation.

***

## Prompt #1: Codebase Analysis & Deep Research

This initial prompt directs the AI to act as a **Deep Research Agent**, focusing on thoroughly understanding the existing codebase before any code is written. [cite_start]It incorporates the ideas of structured analysis, multi-faceted reasoning, and the use of a "structured context memory" inspired by the **Code Researcher** and **TICK** papers[cite: 4, 2463].

```markdown
**Role:** Senior Staff Engineer & Codebase Archaeologist
**Objective:** To conduct a deep and thorough analysis of the codebase at `{codebase_path}` to understand its architecture, key patterns, and historical context. The goal is to produce a clear, concise, and 100% accurate "Codebase Research Summary" that will serve as the foundation for all future work on this project.

**Project Request:** {user_feature_request}

---

### Phase 1: Deep Research & Analysis

You will now perform a multi-step investigation of the codebase. You have access to the following actions. Invoke them sequentially to build a comprehensive understanding. For each step, document your thought process and update the "Structured Context Memory" with your findings.

**Available Actions:**
* [cite_start]`search_definition(symbolName, [filePath])`: Searches for the definition of a specific function, class, struct, or variable[cite: 2557].
* [cite_start]`search_code(regexPattern)`: Performs a regular expression search across the codebase to find usage patterns, function calls, or specific logic[cite: 2559].
* [cite_start]`search_commits(regexPattern)`: Searches commit history (messages and diffs) to understand the evolution of specific components, find justifications for past changes, and identify original authors of relevant code blocks[cite: 2561, 2686].
* `list_files(directory_path)`: Lists all files and subdirectories in a given path.

---

### Task: Build the Structured Context Memory

As you perform your research, populate the following Markdown-formatted "Structured Context Memory". Be methodical. Your summary should be accurate and serve as a reliable guide.

#### Structured Context Memory:

1.  **Core Components & Architecture:**
    * Identify and list the primary directories and their responsibilities.
    * Map out the main data structures and classes. What is their purpose?
    * Trace the primary control flow for a typical operation. How do the major components interact?

2.  **Key Patterns & Conventions:**
    * [cite_start]**Common Patterns:** Identify recurring design patterns (e.g., singleton, factory, observer) or architectural styles (e.g., MVC, microservices)[cite: 2577].
    * [cite_start]**Anti-Patterns:** Note any common anti-patterns or "code smells" you observe that might be relevant to the new feature[cite: 2579].
    * **State Management:** How is application state managed?
    * **Concurrency:** What are the patterns for handling concurrent operations?
    * **Error Handling:** Describe the strategy for error and exception handling.

3.  **Historical Context & Rationale:**
    * Use `search_commits` to find the rationale behind the key components you've identified. Why were they built this way?
    * Identify any major refactoring efforts or architectural shifts in the project's history.

4.  **Initial Hypothesis for Integration:**
    * Based on your research, where do you hypothesize the new feature (`{user_feature_request}`) will most likely integrate? List the key files and functions that seem most relevant.

---

Begin your step-by-step analysis now. For each step, state the action you are taking, the reason for it, and the new information learned.
```

***

## Prompt #2: Best Practices Research Plan

This prompt uses the summary from the first stage to guide the AI in creating a plan for external research. [cite_start]It leverages concepts of **metacognition**, **explicit tool use**, and **decomposing complex questions** from the research on prompt engineering[cite: 650, 660].

```markdown
**Role:** Research Strategist
**Objective:** Using the "Codebase Research Summary" you previously created, develop a "Best Practices Research Plan." This plan will outline the specific questions you need to answer and the search queries you will use to find the best-in-class approaches for implementing `{user_feature_request}` within the identified architectural context.

---

### Phase 2: Formulating the Research Plan

Reference the **Structured Context Memory** you generated. Based on that document and the user's request, create a plan to research the optimal implementation strategy.

#### Your Task:

1.  **Deconstruct the Problem:** Break down the core challenge of `{user_feature_request}` into a series of specific, answerable questions. These questions should bridge the gap between the current state of the codebase and the desired new feature.

2.  **Formulate Search Queries:** For each question, devise the precise search queries you will use to find answers. Your queries should be designed to uncover libraries, frameworks, design patterns, and expert opinions.

3.  **Justify Your Approach:** For each query, provide a brief justification explaining *why* this information is necessary and how it relates to the findings in your "Codebase Research Summary."

---

### Deliverable: Best Practices Research Plan

Please format your output as follows:

#### Research Question 1: {e.g., What is the most robust way to handle asynchronous data streams in a C++ environment using Boost.Asio?}
* **Justification:** The codebase summary indicates that the project heavily relies on Boost.Asio for networking. The new feature requires processing real-time data, so understanding the best practices for asynchronous operations within this existing framework is critical to ensure performance and maintainability.
* **Search Queries:**
    * `"Boost.Asio C++ asynchronous stream processing best practices"`
    * `"high performance networking C++ Boost.Asio examples"`
    * `"error handling strategies for Boost.Asio async_read"`

#### Research Question 2: {e.g., How to implement a thread-safe singleton for logging in a multi-threaded application?}
* **Justification:** ...
* **Search Queries:**
    * ...

---

Proceed with creating the "Best Practices Research Plan."
```

***

## Prompt #3: Step-by-Step Implementation Plan

This prompt synthesizes all prior research into a detailed, executable plan. [cite_start]It uses the principles of **structured guidance**, **decomposition**, and **in-context examples** highlighted as effective prompting techniques[cite: 1728, 1954].

```markdown
**Role:** Principal Architect
**Objective:** Synthesize the "Codebase Research Summary" and the "Best Practices Research Plan" to create a granular, step-by-step "Implementation Blueprint" for the feature: `{user_feature_request}`.

---

### Phase 3: Creating the Implementation Blueprint

You are to create a detailed, actionable plan broken down into logical and sequential tasks. This blueprint will be the direct guide for the implementation phase.

**Additional Context & Constraints:**
* **(Optional) Exemplar Code:** Here is an example of a class that follows our team's style guide for dependency injection and logging. Please adhere to this pattern. `{Insert exemplar code snippet here}`.
* **Definition of Done:** Each task is only considered "done" when it is implemented, has a corresponding unit test, and the test passes.

---

### Your Task:

Create a list of implementation tasks. Each task should be a small, logical unit of work. For each task, specify the following:

* **Task ID:** A unique identifier (e.g., `FEAT-01`, `TEST-01`).
* **Description:** A clear and concise explanation of what needs to be done.
* **Files to Modify:** A list of the file(s) that will be created or changed for this task.
* **Dependencies:** Any other Task IDs that must be completed before this one can start.
* **Acceptance Criteria:** A brief description of what success looks like (e.g., "The `DataProcessor` class correctly parses the input stream and extracts the header.").

---

### Deliverable: Implementation Blueprint

Format your output in a structured way, for example:

| Task ID  | Description                                        | Files to Modify            | Dependencies | Acceptance Criteria                                 |
| :------- | :------------------------------------------------- | :------------------------- | :----------- | :-------------------------------------------------- |
| `FEAT-01`  | Create the `DataParser` interface.                 | `src/interfaces/IParser.h` | None         | Interface defines `parse()` and `getResult()` methods. |
| `FEAT-02`  | Implement the `JSONParser` class based on `IParser`. | `src/parsers/JSONParser.cpp` | `FEAT-01`    | `parse()` method correctly handles valid JSON.      |
| `TEST-01`  | Write unit tests for `JSONParser`.                 | `tests/test_JSONParser.cpp`  | `FEAT-02`    | Tests cover valid input, invalid input, and edge cases. |
| ...      | ...                                                | ...                        | ...          | ...                                                 |

---

Proceed with generating the "Implementation Blueprint."
```

***

## Prompt #4: Plan Execution & Iterative Development

This final prompt guides the AI through the hands-on coding process, emphasizing a **test-driven**, **iterative**, and **self-correcting** workflow. [cite_start]It operationalizes concepts from **Self-Refine**, **STICK**, and the validation phases of agentic workflows[cite: 1600, 8].

```markdown
**Role:** Lead Developer
**Objective:** Execute the "Implementation Blueprint" step-by-step to build, test, and version control the new feature. Adhere strictly to a test-driven development (TDD) and self-refinement cycle.

---

### Phase 4: Implementation, Testing, & Version Control

You will now execute the plan you created. For each task in the "Implementation Blueprint," you must follow this strict workflow loop:

#### Workflow Loop (for each Task ID):

1.  **Announce Task:** State the `Task ID` and `Description` you are beginning to work on.

2.  **Write Code / Write Test:**
    * If the task is a feature (`FEAT-XX`), write the implementation code first.
    * If the task is a test (`TEST-XX`), write the test code.

3.  **Run Test & Evaluate:**
    * Execute the relevant test(s) for the component you are working on.

4.  **Self-Refinement Loop (if tests fail):**
    * **Analyze Failure:** If a test fails, analyze the error message and the code. State your hypothesis for the cause of the failure.
    * **Refine Code:** Modify the code to fix the bug.
    * **Re-run Test:** Go back to step 3. Continue this loop until all relevant tests pass. [cite_start]This is a self-correction process; do not proceed until the code is working[cite: 1903, 168].

5.  **Commit to Version Control (on success):**
    * Once the tests for the task pass, execute a `git commit`.
    * The commit message must be clear and reference the `Task ID`. Example: `git commit -m "feat(parser): Implement JSONParser class - FEAT-02"`

6.  **Proceed to Next Task:** Move to the next `Task ID` in the blueprint and repeat the workflow loop.

---

Begin execution of the "Implementation Blueprint" now. Announce your first task.
```