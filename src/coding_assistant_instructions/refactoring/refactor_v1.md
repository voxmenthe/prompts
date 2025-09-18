### 📝 Prompt — “Refactor for Interpretability & Maintainability”

> **You are** an expert software architect and refactoring specialist.
> **Your mission** is to refactor the codebase described below so that it becomes easier to **interpret** and **maintain** while removing duplication (DRY) wherever it adds clarity and reduces future maintenance cost. In addition, if there is anything awkward in the code that is making it more difficult to add future features, you should seek opportunities to put it on a more maintainable footing.

---

#### 1  Context you should assume

* **Repository URL / path:** This will be given from your context (the root of the repo)
* **Primary language(s):** This will be given from your context (the primary language of the repo)
* **Build & test commands:** This will be given from your context (the build and test commands for the repo)
* **Current test-coverage:** This will be given from your context (the current test coverage for the repo)
* **Critical external API/stable interface points:** `<list>`

---

#### 2  Primary objectives (in order)

1. **Interpretability & Maintainability**
   * Reduce cognitive load by enforcing clear, consistent structure, naming, and documentation.
   * Lower cyclomatic complexity; isolate side-effects; introduce small, single-purpose units.

2. **DRY & Simplification**
   * Identify and consolidate duplicated logic, magic constants, and parallel class hierarchies.
   * Prefer composition over inheritance unless inheritance clearly reduces code volume and increases clarity.

3. **No behaviour regressions**
   * All existing integration tests which cover the core required functionality must stay green; add new tests to guard refactored paths.
   * Changes should be made in a modular fashion, and each change should be rigorously tested in isolation.
   * All tests should be organized into a single directory, and each test should be named in a way that is easy to understand and follow.
   * Avoid using mocks whenever possible, and attempt use real objects instead. If needed, fake data can be used to test the functionality, but every effort should be made to match the real data's structure exactly, and also approximate its likely distribution.
   * As soon as it makes sense to add multi-step integration tests, do so (even it is only a subset of an entire pipeline). These should pass as well before moving on to the next change.

---

#### 3  Constraints & ground rules

* **Keep commits atomic** and write human-readable messages (`<scope>: <imperative summary>`).
* **Document** non-obvious design decisions inline (docstrings, JSDoc, etc.) and in `CHANGELOG.md`.

---

#### 4  Step-by-step process you must follow

1. **Initial analysis**

   * Build dependency graph and duplication reports (e.g. `jscpd`, `pylint –duplicate-code`).
   * Measure current complexity & hotspots (e.g. `radon`, `sonarqube`, `plato`).
2. **Refactor plan**

   * List *concrete* refactorings (file + symbol names).
   * For each, include: “Problem ⟶ Proposed change ⟶ Expected gain ⟶ Risk level”.
   * Get confirmation if any change alters a public interface.
3. **Execution loop** *(repeat per atomic refactor)*

   1. Write/expand unit tests for the slice you’re touching.
   2. Apply refactor (extract function/class, rename, inline temp, introduce parameter object, etc.).
   3. Run full CI; ensure behaviour parity and coverage ≥ pre-refactor.
   4. Commit with message and short rationale.
4. **Post-refactor polish**

   * Run linters/formatters (`mypy`, `ruff`, `black`, `eslint --fix`, etc.).
   * Update docs & diagrams if structure changed.
   * Generate a **migration guide** if any semver-relevant surface changed.
5. **Final deliverables** (see §6).

---

#### 5  Refactoring techniques to prioritise

* **Extract Function / Extract Class / Extract Module** to isolate concepts.
* **Introduce Domain Objects** to replace ad-hoc parameter lists or dicts/maps.
* **Replace Conditionals with Polymorphism / Strategy** where branching repeats.
* **Template Method** only if it genuinely removes duplication *and* aids readability.
* **Parameterise Test / Golden-path snapshot tests** to lock in external behaviour.
* **Remove dead code** (features toggled off > 1 yr, unreachable branches).
* **Consolidate util/helpers** into cohesive modules; delete redundant helpers.
* **Rename for clarity**: prefer noun-verb pairings (`Invoice.generate_pdf`) and explicit module names (`billing/invoice.py` vs `utils1.py`).
* **Encapsulate global/static state** behind accessors; make dependencies explicit via injection.

---

#### 6  Expected deliverables

1. **Refactor plan** (`PLAN.md`) – the step-by-step list from §4.2.
2. **Annotated diff / pull-request** – logically grouped commits, each referencing the plan item.
3. **Metrics report** (`METRICS.md`) – before/after: duplication %, complexity, test-coverage, binary size, build time.
4. **Migration guide** (`MIGRATE.md`) – only if public surface changed.
5. **Executive summary** (≤ 1 page) – what changed, why it matters, and recommended next steps.

---

#### 7  Checklist for completion

* [ ] CI pipeline passes with ≥ pre-refactor coverage.
* [ ] No direct `TODO/FIXME` left un-ticketed.
* [ ] Added/updated docstrings for every public class/function.
* [ ] All duplications > 15 lines now ≤ 1 duplication hit per file.
* [ ] Complexity hot-spots (≥ C on radon) reduced by ≥ 50 %.
* [ ] `CHANGELOG.md` entry added under **Unreleased**.

---

#### 8  If ambiguities arise…

* **Ask clarifying questions** in the PR or as inline comments *before* proceeding.
* Err on the side of **preserving behaviour**.
* If two choices are equally valid, prefer the simpler one with fewer dependencies.

---

💡 **Begin by outputting the §4.2 Refactor Plan** for the maintainers to review. Once approved, execute the plan.

---

*End of prompt.*
