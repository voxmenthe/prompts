# Comprehensive Implementation Planning - Creating the Implementation Blueprint
**Role:** Principal Architect
**Objective:** Synthesize the "Codebase Research Summary" (in `codebase_overview.md`) and the "Best Practices Research Plan" (in `research_plan.md`) to create a granular, step-by-step "Implementation Blueprint" for the feature: `{user_feature_request}`.

You are to create a detailed, actionable plan broken down into logical and sequential tasks. This blueprint will be the direct guide for the implementation phase.

Using the codebase analysis (in `codebase_overview.md`) and research findings (in `research_plan.md`), create a detailed, step-by-step implementation plan for the requested feature. This plan will guide the actual implementation phase.

* **Definition of Done:** Each task is only considered "done" when it is implemented, has a corresponding unit test, and the test passes

## Context:
- **Codebase Analysis**: [From `codebase_overview.md`]
- **Research Findings**: [From `research_plan.md`]
- **Feature Requirements**: [User's request]
- **Additional Resources**: [Any example code, guidelines, or specifications provided]

## Implementation Planning Process:

### Phase 1: Architectural Design
1. **High-Level Design**
   - Create a conceptual design that fits within existing architecture
   - Define new components and their responsibilities
   - Map interactions with existing components
   - Identify any necessary architectural changes

2. **Detailed Component Design**
   For each new or modified component:
   - Define interfaces and contracts
   - Specify data models and structures
   - Design error handling strategies
   - Plan for extensibility and maintenance

### Phase 2: Task Decomposition
3. **Break Down Implementation into Tasks**
   Apply these principles:
   - Each task should be independently testable
   - Tasks should follow dependency order
   - Group related changes together
   - Size tasks for 1-4 hours of work each

4. **Task Categorization**
   Organize tasks into:
   - **Foundation**: Core infrastructure changes
   - **Core Features**: Main functionality implementation
   - **Integration**: Connecting with existing system
   - **Polish**: UI/UX improvements, optimization
   - **Documentation**: Code docs, user guides

### Phase 3: Implementation Sequencing
5. **Dependency Analysis**
   - Map task dependencies
   - Identify critical path
   - Plan for parallel work where possible
   - Build in checkpoints for validation

6. **Risk-Based Ordering**
   Prioritize tasks that:
   - Validate core assumptions early
   - Have highest technical risk
   - Block other work if delayed
   - Could reveal need for design changes

### Phase 4: Testing Strategy
7. **Test Planning for Each Task**
   Define for each implementation task:
   - Unit test requirements
   - Integration test needs
   - Edge cases to cover
   - Performance benchmarks (if applicable)

8. **Continuous Validation Plan**
   - Define acceptance criteria for each task
   - Plan incremental integration tests
   - Identify regression test requirements
   - Set up monitoring/logging checkpoints

**Additional Notes on Tasks:**

Each task should be a small, logical unit of work. For each task, specify the following:

* **Task ID:** A unique identifier (e.g., `FEAT-01`, `TEST-01`).
* **Description:** A clear and concise explanation of what needs to be done.
* **Files to Modify:** A list of the file(s) that will be created or changed for this task.
* **Dependencies:** Any other Task IDs that must be completed before this one can start.
* **Acceptance Criteria:** A brief description of what success looks like (e.g., "The `DataProcessor` class correctly parses the input stream and extracts the header.").

## Deliverable
`implementation_plan.md` containing:

1. **Back‑casted success snapshot** – describe repository state *after* the feature ships (reverse‑thought anchor).&#x20;
2. **Work‑breakdown structure (WBS)**
   * Each task <= 4 hours dev time, includes ref links to files.
3. **Task metadata** - some things that are useful for tracking progress and managing the implementation process.
4. **Integration test storyboard** – high‑level user journeys & edge cases.
5. **Quality gates** – lint, security scan, performance budgets.
6. **Milestone timeline** – ordered by critical path; highlight any parallelisable tasks.
7. **Review checklist** – what code reviewers must confirm.

## Prompt guidelines

* Pre‑pend each task with **“WHY”** (one‑line rationale) to make intent explicit (clarity/objectives).&#x20;
* Select example commit messages for two typical tasks (demo principle).&#x20;
* Keep total length ≈ 800‑1200 tokens (token‑quantity balance).&#x20;


## Output Format:
Structure your implementation plan as follows:

```markdown
# Feature Implementation Plan

## 1. Architectural Overview
### Design Summary
[High-level description of how feature fits into existing architecture]

### Component Diagram
[ASCII or description of component relationships]

### Key Design Decisions
- **Decision 1**: [What and why]
- **Decision 2**: [What and why]

## 2. Implementation Tasks

### Phase 1: Foundation (Est. X hours)
#### Task 1.1: [Task Name]
- **Description**: [What needs to be done]
- **Files to Modify/Create**: 
  - `path/to/file1.ext` - [Purpose]
  - `path/to/file2.ext` - [Purpose]
- **Dependencies**: [None or list other tasks]
- **Test Requirements**:
  - Unit test for [functionality]
  - Edge case: [description]
- **Acceptance Criteria**:
  - [ ] [Criterion 1]
  - [ ] [Criterion 2]
- **Git Commit Message Template**: `feat(module): Add [specific change]`

[Repeat for each task]

### Phase 2: Core Features (Est. Y hours)
[Continue with tasks]

## 3. Testing Strategy
### Unit Testing Plan
- **Coverage Goal**: X%
- **Key Test Scenarios**: [List]

### Integration Testing Plan
- **Test Environment Setup**: [Description]
- **Critical Paths to Test**: [List]

## 4. Implementation Schedule (logical chunk based - not time based)
### Logical Chunk #1
- [ ] Tasks 1.1 - 1.3
- [ ] Integration checkpoint

### Logical Chunk #2
- [ ] Tasks 2.1 - 2.4
- [ ] Performance validation

[Continue schedule]

## 5. Risk Mitigation
### Identified Risks
1. **Risk**: [Description]
   - **Mitigation**: [Approach]
   - **Contingency**: [Backup plan]

## 6. Success Metrics
- **Functional**: [List criteria]
- **Performance**: [Benchmarks]
- **Quality**: [Code metrics]
```

## How to Think:

## Reverse Reasoning Validation:
Starting from the desired end state:
1. What does the fully implemented feature look like?
2. What must be in place for that to work?
3. What dependencies must those components have?
4. Continue working backward to current state

## Self-Verification Checklist:
- [ ] All research findings are incorporated into the plan
- [ ] Each task is atomic and independently valuable
- [ ] Dependencies are clearly mapped and logical
- [ ] Test requirements are comprehensive
- [ ] The plan accommodates discovered codebase constraints
- [ ] Risk mitigation strategies are practical
- [ ] Timeline is realistic given task complexity

## Consider each of the following:
- Have I considered both the "happy path" and edge cases?
- Is each task small enough to complete and test in one session?
- Have I planned for integration points that might reveal issues?
- Does the sequence allow for early validation of assumptions?

Remember: This plan is your roadmap. It should be detailed enough to guide implementation but flexible enough to accommodate discoveries made during coding.
