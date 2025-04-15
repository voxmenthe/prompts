
The following are practical, hard-won wisdom from a senior developer for maintaining long-running, scalable, and maintainable Typescript code. Please provide a lightly edited version of the following principles in exactly the same style and tone and orientation as the examples below, adding any key missing pieces and tricks that will really make the difference for large and complex codebases. Remember, you are writing for a senior developer who has been in the trenches for years and sharing practical advice, not textbook principles. Remember to focus on tricks and principles and try to avoid suggesting extra dependencies or packages or anything that adds extra baggage. Also make sure to check the principles below for any inconsistencies or any instructions that might be difficult to follow or overly cumbersome. Provide your revised version of the complete set of principles below.

* Always write comments for what you are doing, major assumptions, and warnings, before you write the rest of the code.
* Print informative console logs frequently with stage prefixes, timestamps and color codes.
for Typescript code
* Keep the main (e.g., index.ts) file small readable.
- Try to separate types into a types.ts file if it makes sense.
- Try to separate small functional utilities into a utils.ts file if it makes sense.
- Try to separate large classes into their own files if it makes sense.

* When you handle errors, beyond just printing the error message, print out the states of the variables that led to the error, so that we can debug things like "[ERROR] Failed to process file: TypeError: Cannot read properties of undefined"

For writing new long running/expensive code (ONLY NEW CODE - don't apply when modifying existing code):
### ✅ **1. Unbundle run() into: init(), continue(id), cleanup(id) functions**. 
- Never assume these will always be called in order. we may need to resume continue or cleanup after a failure anywhere in the workflow.
- catch failures and capture as much state as possible after a failure checkpoint when they happen for debugging, including error stack/message, timing, etc.

### ✅ **2. Always checkpoint (save state into files/databases) and resume from id's**. 
Between major modules, only work from id's, don't pass classes or other states. This forces you to keep things serializable and therefore loggable, reproducible, parallelizable.
- try not to name your ID's "id" if you can add extra detail like "runId", "taskId", "subTaskId".
- each checkpoint should be saved with a starting timestamp (UTC) + unique id prefix (simpleSk2Id, substring mds of a random number - that serves as a run id), and overwritten as the run progresses so as to preserve the latest state in case the run fails (put checkpointing code into its own file)
- store final results in a different folder than checkpoint (don't forget to create the folders if they don't exist, or use a system boundary that does that, or you will have errors)

### ✅ **3. Pay attention to system boundaries**
Any time you cross a system boundary (eg. call microservices or external API - and yes this includes LLM apis meaning every ai project needs this), **implement RATE LIMITS, TIMEOUTS, RETRIES, LOG TRACES (track start-end time of traces)**. Make reasonable assumptions for values based on the context and make it easy for the user to tweak, and consider how to gracefully handle errors when all that fails.

---

### ✅ **4. Never mix responsibilities in utility files**
- If your `utils.ts` file starts containing both business logic and pure functions, split it clearly into files named by domain (e.g., `fileUtils.ts`, `stringUtils.ts`, `timeUtils.ts`).
- A well-organized set of small, self-describing utility files is invaluable over time.

---

### ✅ **5. Treat timestamps as first-class citizens**
- Always store timestamps explicitly in UTC and use ISO 8601 format (`2025-04-01T12:01:22Z`).
- Clearly label all variables that represent timestamps (`startedAtUtc`, `completedAtUtc`), never just `time` or `date`.

---

### ✅ **6. Never silently swallow exceptions**
- If you ever find yourself writing `try/catch` with empty catch blocks, pause immediately and add at least a logging statement.
- Hidden exceptions are debugging nightmares.

---

### ✅ **7. Favor explicit data shapes**
- Strongly avoid overly generic types like `any`, `object`, or `Record<string, any>`. Even a single-use interface can prevent subtle bugs months later.
- When you checkpoint states, explicitly version the data shape (e.g., include `"checkpointVersion": "v1"`), which prevents confusion when later loading older checkpoints.

---

### ✅ **8. Always put boundaries around external dependencies**
- Wrap all external API calls (including database queries, LLM APIs, etc.) in a dedicated class or function clearly labeled as a "Gateway," "Client," or "Connector".
- This makes debugging, mocking, rate-limiting, and handling timeouts vastly simpler.

---

### ✅ **9. Code defensively around resuming long tasks**
- Don’t assume resumed tasks always continue with the same config or environment. Validate assumptions explicitly at startup of every `continue()` or `cleanup()`.
- Immediately fail fast if assumptions have changed, rather than silently continuing.

---

### ✅ **10. Don’t leak sensitive details to logs**
- Logs should be thorough, but never include raw user credentials, secret keys, tokens, or other sensitive details.
- Always sanitize logs (e.g., mask tokens as `********abcdef`) at the moment of logging.

---

### ✅ **11. Checkpoints should be human-readable**
- If your serialized checkpoint data can’t easily be inspected by a human (too large, compressed, binary, etc.), consider also storing a concise summary alongside it (e.g., `summary.json`).
- This allows rapid debugging by humans without custom scripts to decode your checkpoints.

---

### ✅ **12. Avoid boolean flags for readability**
- Instead of `run(true, false)` prefer explicit parameters like `run({shouldCheckpoint: true, verboseLogging: false})`.
- Named parameters are instantly readable 12 months later.

---

### ✅ **13. Don't trust default behaviors implicitly**
- If you rely on defaults (timeouts, retry counts, etc.), explicitly document and justify these defaults in comments. 
- Defaults are hidden assumptions. Exposing them clearly prevents bugs during future scaling or refactoring.

---

### ✅ **14. Fail clearly when system boundaries misbehave**
- Clearly identify and log boundary failures (timeouts, rate-limits exceeded, network errors) separately from internal logic failures.
- Explicit labels in logs like `[EXTERNAL_FAILURE]` or `[INTERNAL_FAILURE]` make incident triage much faster.

---

### ✅ **15. Centralize and strictly isolate mutable state**
- Keep mutable state centralized and clearly identified in a single, explicit "state" object or class.
- Do **not** scatter mutable variables across many modules.  
- Always mutate state through dedicated methods or helpers, not arbitrarily or directly.
- Enforce serialization/deserialization explicitly (to JSON or DB) at state boundaries. This guarantees reproducibility, simplifies debugging, and prevents accidental state corruption.

Example pattern to follow:

```typescript
// stateManager.ts
type RunState = {
  runId: string;
  startedAtUtc: string;
  currentStep: string;
  progress: number;
  metadata?: Record<string, unknown>;
};

let state: RunState | null = null;

export const initState = (initialState: RunState) => {
  state = initialState;
};

export const updateProgress = (step: string, progress: number) => {
  if (!state) throw new Error("State not initialized");
  state.currentStep = step;
  state.progress = progress;
};

export const getStateSnapshot = (): RunState => {
  if (!state) throw new Error("State not initialized");
  return JSON.parse(JSON.stringify(state)); // ensures serializability
};
```

### ✅ **16. Always explicitly type function return values**
- Don’t rely on Typescript’s type inference for complex functions—explicitly annotate function returns.
- Explicit return types help catch subtle bugs early, clarify intent, and greatly enhance maintainability.

---

### ✅ **17. Favor consistent naming conventions for error codes/messages**
- Standardize prefixes for different categories of errors: `[DB_ERROR]`, `[VALIDATION_ERROR]`, `[NETWORK_ERROR]`, etc.
- This uniformity speeds up debugging, filtering logs, and incident response.

---

### ✅ **18. Validate configuration at startup rigorously**
- At the beginning of your main entry (`index.ts`), explicitly check and validate all required environment variables or configuration files.
- Immediately exit with a clear error message if a required config is missing or invalid—preventing cryptic runtime failures.

---

### ✅ **19. Use structured logging rather than raw strings**
- Prefer structured logging objects (`{ runId, timestampUtc, message, details }`) instead of concatenated log strings.
- Structured logs can easily be parsed, filtered, and analyzed automatically, drastically reducing operational friction.

---

### ✅ **20. Avoid deeply nested logic**
- If your function logic exceeds 3 levels of nesting, pause and refactor into smaller, named helper functions.
- Clear linear logic flows greatly simplify debugging and reduce cognitive load when revisiting code later.

---

### ✅ **21. Clearly define retry strategies at external boundaries**
- Implement explicit retry logic with exponential backoff and maximum retry limits around external system calls.
- Explicitly log each retry attempt, including backoff duration, attempt number, and the triggering error.

---

### ✅ **22. Limit side-effects to explicit layers**
- Clearly separate pure functions from side-effectful operations (filesystem writes, network calls, DB updates).
- Never embed hidden side-effects inside utility or helper methods; always explicitly name these functions (`writeCheckpoint`, `updateDatabaseRecord`).

---

### ✅ **23. Always document implicit contracts between modules**
- If modules implicitly depend on states or side-effects set by other modules, explicitly document this dependency in comments.
- Undocumented assumptions cause hard-to-trace bugs when scaling or modifying behavior months later.

---

### ✅ **24. Enforce explicit lifecycle management for resources**
- Resources such as open connections, file handles, or timers should have explicit `open()` and `close()` methods.
- Clearly document the ownership and lifecycle expectations (who calls close, when, and under what circumstances).

---

### ✅ **25. Limit global constants to a single module (`constants.ts`)**
- Centralize all shared numeric/string constants (timeouts, retry limits, file paths, etc.) to one clearly named file.
- Avoid spreading "magic numbers" or strings throughout the codebase.

---

### ✅ **26. Prefer throwing custom error types**
- Define and use custom error classes (`class DatabaseConnectionError extends Error`) instead of generic `Error` objects.
- Custom error types provide semantic clarity and simplify handling logic downstream.

---

### ✅ **27. Explicitly name intermediate transformations**
- Rather than chaining multiple transformations inline (`data.filter(...).map(...).reduce(...)`), assign intermediate results to clearly named constants.
- Explicit intermediate steps make debugging and refactoring significantly easier.

---

### ✅ **28. Always explicitly handle promise rejections**
- Never let promises silently reject; always attach `.catch()` handlers or use `await` with explicit try-catch blocks.
- Unhandled rejections lead to subtle, difficult-to-debug production issues.

---

### ✅ **29. Be explicit with state transitions**
- Clearly document and enforce allowed transitions between states (e.g., `initialized` → `running` → `completed`).
- Reject and log attempts at illegal state transitions clearly—this avoids silent corruptions or hard-to-debug race conditions.

---

### ✅ **30. Centralize error handling logic for system boundaries**
- Keep centralized error-handling logic for external system interactions (DB errors, network failures, API responses).
- Centralized handlers simplify debugging, retries, and standardize logs for quick incident resolution.