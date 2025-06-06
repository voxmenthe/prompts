# Task Implementation & Execution

This document defines rules for task creation, execution, and tracking, ensuring all work is broken down into manageable, auditable units while enabling AI agents to contribute effectively.

## 1. Task Documentation Standards

### 1.1 File Structure
- **Location Pattern**: `docs/delivery/<PBI-ID>/`
- **Task List**: `tasks.md`
- **Task Details**: `<PBI-ID>-<TASK-ID>.md` (e.g., `1-1.md`)

### 1.2 Required Task Sections
```markdown
# [Task-ID] [Task-Name]
## Description
## Status History
## Requirements
## Implementation Plan
## Verification
## Files Modified
## AI Agent Notes  # New section for AI insights
```

### 1.3 Task Creation Rules
1. Each task must have its own dedicated markdown file
2. Task files must be created immediately when added to the index
3. Tasks must be linked using pattern: `[description](./<PBI-ID>-<TASK-ID>.md)`
4. Individual task files must link back: `[Back to task list](../tasks.md)`

## 2. Task Granularity and Scoping

### 2.1 Task Size Guidelines
Tasks must be as small as practicable while remaining cohesive:
- **Micro-tasks** (1-2 hours): Configuration, constants, simple utilities
- **Small tasks** (2-4 hours): Single feature implementation, focused refactoring
- **Medium tasks** (4-8 hours): Integration work, complex features
- **Large tasks** (8+ hours): Should be split into smaller tasks

### 2.2 AI Agent Task Breakdown
When creating tasks, the AI_Agent should:
1. Suggest optimal task granularity based on complexity
2. Identify natural boundaries between tasks
3. Ensure each task has a clear, testable outcome
4. Flag when tasks are too large or too small

## 3. Task Workflow and Status Management

### 3.1 Status Definitions
- **Proposed**: Initial state of newly defined task
- **Agreed**: User has approved task description and priority
- **InProgress**: AI Agent is actively working on task
- **Review**: Implementation complete, awaiting validation
- **Done**: User has reviewed and approved implementation
- **Blocked**: Cannot proceed due to external dependency

### 3.2 Status Synchronization
1. **Immediate Updates**: Update both task file and index in same commit
2. **Status History**: Always add entry when changing status
3. **Status Verification**: Check both locations before starting work
4. **Mismatch Resolution**: Update to most recent status immediately

### 3.3 One Task In Progress Rule
- Only one task per PBI should be InProgress at any time
- Exceptions require explicit User approval
- AI_Agent should queue next task recommendations

## 4. Implementation Workflow

### 4.1 Pre-Implementation Phase (Proposed → Agreed)
The AI_Agent should:
1. Analyze task requirements thoroughly
2. Identify potential implementation approaches
3. Document estimated complexity and risks
4. Suggest related tasks that might be needed
5. Create implementation plan with:
   - Technical approach options
   - Estimated time/complexity
   - Potential blockers or dependencies

### 4.2 Implementation Phase (Agreed → InProgress)
1. Verify no other tasks are InProgress for the PBI
2. Create feature branch if using version control
3. AI_Agent should:
   - Document implementation decisions in real-time
   - Flag any deviations from plan immediately
   - Suggest optimizations within scope
   - Update "Files Modified" section progressively

### 4.3 Code Generation Best Practices
The AI_Agent must:
1. Follow project coding standards
2. Include appropriate comments and documentation
3. Implement comprehensive error handling
4. Use descriptive variable and function names
5. Apply DRY principle consistently
6. Create constants for all repeated values

### 4.4 Review Preparation (InProgress → Review)
1. Ensure all requirements are met
2. Run all relevant tests
3. AI_Agent should prepare:
   - Summary of changes made
   - Test results overview
   - Any deviations from original plan
   - Suggested improvements for future tasks

## 5. Task Implementation Patterns

### 5.1 Common Task Types

#### Configuration Tasks
- Focus on correctness and documentation
- AI should suggest validation approaches
- Include migration considerations

#### Feature Implementation
- Start with interface/contract definition
- Implement core logic with error handling
- Add comprehensive logging
- Include unit tests

#### Integration Tasks
- Document all external dependencies
- Include retry and fallback logic
- Implement circuit breakers where appropriate
- Add integration tests

#### Refactoring Tasks
- Maintain behavior while improving structure
- Document why refactoring is needed
- Include before/after performance metrics
- Ensure test coverage remains high

### 5.2 AI Agent Value-Add Patterns
For each task type, the AI should:
1. Suggest relevant design patterns
2. Identify potential edge cases
3. Recommend testing strategies
4. Propose monitoring/logging points

## 6. Version Control Integration

### 6.1 Commit Standards
When completing tasks:
```bash
git commit -m "<task_id> <task_description>"
# Example: "1-7 Add pino logging for database debugging"
```

### 6.2 Pull Request Format
- Title: `[<task_id>] <task_description>`
- Description must include:
  - Link to task document
  - Summary of changes
  - Testing performed
  - Any deviations from plan

### 6.3 AI Agent Git Practices
The AI_Agent should:
1. Suggest meaningful commit messages
2. Recommend logical commit boundaries
3. Identify files that should be in .gitignore
4. Flag uncommitted changes before task completion

## 7. Task Completion Criteria

### 7.1 Definition of Done
A task is complete when:
1. All requirements are implemented
2. Code passes all tests
3. Documentation is updated
4. Code review feedback is addressed
5. Status is synchronized in both locations

### 7.2 Post-Completion Review
Before marking Done, review:
1. Next tasks for continued relevance
2. Lessons learned for future tasks
3. Technical debt introduced
4. Opportunities for follow-up improvements

## 8. AI Agent Proactive Assistance

### 8.1 During Implementation
The AI_Agent should proactively:
1. Identify code smells and suggest fixes
2. Recommend performance optimizations
3. Suggest additional error scenarios
4. Flag security considerations
5. Identify missing test cases

### 8.2 Cross-Task Insights
The AI_Agent should:
1. Identify patterns across tasks
2. Suggest shared utilities or abstractions
3. Recommend task order optimizations
4. Flag duplicate effort opportunities

## 9. Error Handling and Recovery

### 9.1 Implementation Errors
When errors occur:
1. Document the error in task history
2. Analyze root cause
3. Propose remediation approach
4. Update implementation plan

### 9.2 Blocked Task Management
When blocked:
1. Document blocking reason clearly
2. Identify workaround possibilities
3. Suggest alternative tasks to progress
4. Estimate unblocking timeline

## 10. Task Index Management

### 10.1 Index File Structure
```markdown
# Tasks for PBI <ID>: <Title>
This document lists all tasks associated with PBI <ID>.
**Parent PBI**: [PBI <ID>: <Title>](./prd.md)

## Task Summary
| Task ID | Name | Status | Description |
| :------ | :--- | :----- | :---------- |
| <ID>    | [<Name>](./<file>.md) | Status | Brief description |
```

### 10.2 AI Agent Index Maintenance
The AI should:
1. Keep index synchronized with task files
2. Flag missing or orphaned task files
3. Suggest task reordering based on dependencies
4. Identify completed tasks ready for archival