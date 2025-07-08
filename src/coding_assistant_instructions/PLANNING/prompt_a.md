Based on the research papers provided, I'll create four comprehensive prompts that incorporate best practices for prompt engineering, including structured reasoning, clear objectives, and metacognitive guidance.

<artifact identifier="prompt-1-codebase-analysis" type="text/plain" title="Prompt 1: Codebase Analysis and Documentation">
# Prompt 1: Comprehensive Codebase Analysis and Documentation

You are an expert software engineer tasked with thoroughly analyzing an existing codebase to prepare for implementing a new feature. Your goal is to create a precise, accurate, and comprehensive documentation of the current state of the codebase.

## Objectives:
1. Analyze the codebase structure, architecture, and key components
2. Document the current functionality and design patterns
3. Identify integration points for the new feature
4. Create a clear, concise summary that will serve as the foundation for planning

## Step-by-Step Analysis Process:

### Phase 1: High-Level Overview
1. **Project Structure Analysis**
   - Map out the directory structure and organization
   - Identify the main entry points
   - Document the build system and dependencies
   - Note any configuration files and their purposes

2. **Architecture Assessment**
   - Identify the architectural pattern (MVC, microservices, monolithic, etc.)
   - Document the technology stack and frameworks used
   - Map out the data flow and system boundaries
   - Identify external dependencies and integrations

### Phase 2: Deep Component Analysis
3. **Core Components Identification**
   - List all major modules/packages and their responsibilities
   - Document the relationships and dependencies between components
   - Identify shared utilities and common patterns
   - Note any design patterns implemented (Factory, Observer, etc.)

4. **Data Layer Analysis**
   - Document the data models and schemas
   - Identify data storage mechanisms (databases, files, caches)
   - Map out data access patterns and ORMs/DAOs
   - Note any data validation and transformation logic

5. **Business Logic Layer**
   - Document core business rules and workflows
   - Identify service layers and their interfaces
   - Map out any state management mechanisms
   - Note error handling and logging patterns

6. **Interface Layer Analysis**
   - Document APIs (REST, GraphQL, RPC, etc.)
   - Identify user interface components and patterns
   - Map out authentication and authorization mechanisms
   - Note any client-server communication patterns

### Phase 3: Integration Point Identification
7. **Feature Integration Analysis**
   - Identify potential integration points for the new feature
   - Document which components will likely need modification
   - Note any architectural constraints or considerations
   - Highlight areas that might require refactoring

### Phase 4: Documentation Synthesis
8. **Create Comprehensive Summary**
   - Synthesize findings into a structured document
   - Include diagrams where helpful (component diagrams, sequence diagrams)
   - Highlight key insights and potential challenges
   - Provide a clear assessment of codebase readiness for new feature

## Output Format:
Please structure your analysis as follows:

```markdown
# Codebase Analysis Summary

## 1. Project Overview
- **Project Name**: [Name]
- **Primary Purpose**: [Brief description]
- **Technology Stack**: [List key technologies]
- **Architecture Pattern**: [Pattern used]

## 2. Directory Structure
[Tree view or structured list of key directories]

## 3. Core Components
### Component A
- **Purpose**: [Description]
- **Dependencies**: [List]
- **Key Files**: [List]

[Repeat for each major component]

## 4. Data Architecture
- **Storage**: [Database/file systems used]
- **Models**: [Key data models]
- **Access Patterns**: [How data is accessed]

## 5. Integration Points for New Feature
- **Recommended Integration Points**: [List with justification]
- **Components Requiring Modification**: [List]
- **Potential Challenges**: [List]

## 6. Key Insights and Recommendations
[Summary of findings and recommendations for feature implementation]
```

## Self-Verification Checklist:
Before finalizing your analysis, verify:
- [ ] All major components have been identified and documented
- [ ] Dependencies between components are clearly mapped
- [ ] Integration points are feasible and well-justified
- [ ] The summary is accurate, concise, and actionable
- [ ] No critical architectural elements have been overlooked

Remember: Accuracy is paramount. If you're uncertain about any aspect, clearly indicate this in your documentation. The goal is to provide a reliable foundation for the implementation planning phase.
</artifact>

<artifact identifier="prompt-2-research-planning" type="text/plain" title="Prompt 2: Best Practices Research Planning">
# Prompt 2: Best Practices Research Planning

Using the codebase analysis from Prompt 1, you will now create a comprehensive research plan to identify the best practices for implementing the requested feature within the context of this specific codebase.

## Context:
- **Codebase Summary**: [Will be provided from Prompt 1 output]
- **Feature to Implement**: [User's requested feature]
- **Your Goal**: Create a targeted research plan that considers both general best practices and codebase-specific constraints

## Research Planning Process:

### Phase 1: Define Research Objectives
1. **Feature Requirements Analysis**
   - Break down the feature into core technical requirements
   - Identify specific technical challenges based on the codebase analysis
   - List knowledge gaps that need to be addressed
   - Prioritize research areas by implementation impact

2. **Contextual Constraints Identification**
   - Note codebase-specific limitations from the analysis
   - Identify which existing patterns must be followed
   - List any architectural decisions that constrain implementation
   - Consider backward compatibility requirements

### Phase 2: Research Strategy Development
3. **General Best Practices Research**
   Create search queries for:
   - Industry-standard implementations of similar features
   - Design patterns applicable to this feature type
   - Performance optimization techniques
   - Security considerations and common pitfalls
   - Testing strategies for this type of feature

4. **Technology-Specific Research**
   Based on the codebase's technology stack, plan searches for:
   - Framework-specific implementation guidelines
   - Language-specific idioms and patterns
   - Compatible libraries and tools
   - Version-specific considerations and deprecations
   - Community-recommended approaches

5. **Integration Pattern Research**
   Plan to investigate:
   - How similar features integrate with the identified architecture pattern
   - Migration strategies if refactoring is needed
   - Dependency injection and coupling strategies
   - API design best practices for the codebase's style

### Phase 3: Research Execution Plan
6. **Prioritized Search Strategy**
   Order your research tasks by:
   - Critical path dependencies
   - Risk mitigation priority
   - Implementation complexity
   - Time sensitivity

7. **Research Validation Criteria**
   For each research area, define:
   - What constitutes a reliable source
   - How to validate conflicting information
   - Criteria for selecting between alternative approaches
   - How to assess applicability to the specific codebase

## Output Format:
Structure your research plan as follows:

```markdown
# Feature Implementation Research Plan

## 1. Feature Breakdown
### Core Requirements:
1. [Requirement 1]
   - Technical Challenge: [Description]
   - Research Priority: [High/Medium/Low]
   - Codebase Constraint: [From analysis]

[Repeat for each requirement]

## 2. Research Areas

### Area A: [e.g., "Authentication Integration"]
**Objective**: [What you need to learn]
**Key Questions**:
- [Question 1]
- [Question 2]

**Search Queries**:
1. "[Proposed search query 1]"
2. "[Proposed search query 2]"

**Validation Criteria**:
- Source must be from [criteria]
- Information must be applicable to [context]

[Repeat for each research area]

## 3. Research Execution Timeline
1. **Phase 1 - Critical Path Research** (Priority: Immediate)
   - [Research Area X]
   - [Research Area Y]

2. **Phase 2 - Integration Research** (Priority: High)
   - [Research Area Z]

[Continue with phases]

## 4. Expected Outcomes
- **Decision Points**: [List key decisions that research will inform]
- **Risk Mitigations**: [How research addresses identified risks]
- **Success Criteria**: [How to know research is complete]
```

## Metacognitive Guidance:
As you create this plan:
- Consider both what you need to know and why you need to know it
- Think about the order of research - some findings may influence others
- Identify which aspects are most critical to get right vs. nice-to-have
- Consider how each research finding will directly impact implementation

## Self-Verification:
Before finalizing, ensure:
- [ ] All technical requirements have corresponding research areas
- [ ] Codebase constraints are reflected in the research approach
- [ ] Search queries are specific and likely to yield relevant results
- [ ] The plan addresses both the "what" and "how" of implementation
- [ ] Risk areas identified in the codebase analysis are addressed

Remember: A well-planned research phase prevents costly implementation mistakes. Focus on actionable, specific research that directly supports implementation decisions.
</artifact>

<artifact identifier="prompt-3-implementation-planning" type="text/plain" title="Prompt 3: Step-by-Step Implementation Planning">
# Prompt 3: Comprehensive Implementation Planning

Using the codebase analysis (Prompt 1) and research findings (Prompt 2), create a detailed, step-by-step implementation plan for the requested feature. This plan will guide the actual implementation phase.

## Context:
- **Codebase Analysis**: [From Prompt 1]
- **Research Findings**: [From Prompt 2]
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

## Output Format:
Structure your implementation plan as follows:

```markdown
# Feature Implementation Plan

## 1. Architectural Overview
### Design Summary
[High-level description of how feature fits into existing architecture]

### Component Diagram
```
[ASCII or description of component relationships]
```

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

## 4. Implementation Schedule
### Week 1
- [ ] Tasks 1.1 - 1.3
- [ ] Integration checkpoint

### Week 2
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

## Metacognitive Prompts:
- Have I considered both the "happy path" and edge cases?
- Is each task small enough to complete and test in one session?
- Have I planned for integration points that might reveal issues?
- Does the sequence allow for early validation of assumptions?

Remember: This plan is your roadmap. It should be detailed enough to guide implementation but flexible enough to accommodate discoveries made during coding.
</artifact>

<artifact identifier="prompt-4-implementation-execution" type="text/plain" title="Prompt 4: Implementation Execution with Testing and Version Control">
# Prompt 4: Systematic Implementation Execution

Execute the implementation plan created in Prompt 3, following test-driven development practices and maintaining version control discipline. Work methodically through each task, creating tests first, implementing features, and committing progress incrementally.

## Context:
- **Implementation Plan**: [From Prompt 3]
- **Current Task**: [Will be updated as you progress]
- **Repository State**: [Current branch, last commit]

## Execution Framework:

### For Each Task in the Plan:

#### Phase 1: Pre-Implementation Setup
1. **Task Preparation**
   - Review the task requirements from the plan
   - Ensure all dependencies are complete
   - Create feature branch: `git checkout -b feature/[task-id]-[brief-description]`
   - Set up any necessary development environment changes

2. **Test-First Development**
   - Write test cases BEFORE implementation
   - Include unit tests for happy path
   - Include edge case tests
   - Include integration tests where applicable
   - Ensure tests fail appropriately (red phase)

#### Phase 2: Implementation
3. **Core Implementation**
   - Implement the minimal code to make tests pass (green phase)
   - Follow the codebase's established patterns
   - Maintain consistent code style
   - Add appropriate error handling
   - Include necessary logging

4. **Refactoring**
   - Refactor for clarity and efficiency (refactor phase)
   - Ensure all tests still pass
   - Check for code duplication
   - Optimize where necessary
   - Update documentation/comments

#### Phase 3: Validation and Integration
5. **Comprehensive Testing**
   - Run all unit tests for the module
   - Run integration tests
   - Perform manual testing if applicable
   - Check for performance impacts
   - Validate against acceptance criteria

6. **Code Review Preparation**
   - Self-review all changes
   - Ensure code follows project conventions
   - Check test coverage metrics
   - Update relevant documentation
   - Prepare clear commit message

#### Phase 4: Version Control
7. **Git Workflow**
   ```bash
   # Stage changes selectively
   git add -p
   
   # Commit with descriptive message
   git commit -m "feat(module): Brief description
   
   - Detailed point 1
   - Detailed point 2
   
   Implements: #task-id"
   
   # Push to feature branch
   git push origin feature/[task-id]-[brief-description]
   ```

### Progress Documentation Template:

After each task completion, document:

```markdown
## Task Completion Report: [Task ID] - [Task Name]

### Implementation Summary
- **Start Time**: [Timestamp]
- **Completion Time**: [Timestamp]
- **Branch**: `feature/[task-id]-[description]`
- **Commit Hash**: [hash]

### What Was Implemented
[Brief description of the implementation]

### Tests Created
1. **Test File**: `path/to/test_file.ext`
   - Test case: [Description] - Status: ✅
   - Test case: [Description] - Status: ✅

### Code Changes
- **Files Modified**:
  - `path/to/file1.ext`: [What changed]
  - `path/to/file2.ext`: [What changed]
- **Files Added**:
  - `path/to/newfile.ext`: [Purpose]

### Challenges Encountered
- **Challenge**: [Description]
  - **Solution**: [How it was resolved]

### Integration Notes
- **Dependencies Updated**: [If any]
- **Breaking Changes**: [None or description]
- **Migration Required**: [Yes/No, details if yes]

### Next Steps
- [ ] Merge to main branch after review
- [ ] Update documentation if needed
- [ ] Notify team of any interface changes

### Metrics
- **Test Coverage**: [X%]
- **Tests Passed**: [X/Y]
- **Code Lines**: +[Added] -[Removed]
```

## Continuous Monitoring Checklist:
During implementation, continuously verify:

- [ ] Tests are written before code
- [ ] All tests pass before moving to next task
- [ ] Code follows established patterns
- [ ] No regression in existing functionality
- [ ] Performance benchmarks are met
- [ ] Security considerations are addressed
- [ ] Documentation is updated alongside code

## Metacognitive Checkpoints:
After each task, ask yourself:
1. Does this implementation fully satisfy the task requirements?
2. Have I introduced any technical debt?
3. Is the code maintainable and understandable?
4. Have I missed any edge cases?
5. Would another developer understand my changes?

## Error Recovery Protocol:
If issues arise:
1. **Don't Panic**: Revert to last known good state if needed
2. **Analyze**: What assumption was incorrect?
3. **Communicate**: Document the issue for plan adjustment
4. **Adapt**: Modify approach based on new information
5. **Learn**: Update tests to catch similar issues

## Daily Progress Summary:
At the end of each work session:
```markdown
### Daily Summary - [Date]
- **Tasks Completed**: [List]
- **Blockers Encountered**: [List or None]
- **Plan Adjustments Needed**: [Description or None]
- **Tomorrow's Priority**: [Next task(s)]
- **Overall Progress**: [X%] of implementation plan
```

## Self-Verification Protocol:
Before considering any task complete:
- [ ] All tests pass
- [ ] Code coverage meets project standards
- [ ] No linting or style errors
- [ ] Documentation is complete
- [ ] Commit message follows conventions
- [ ] Integration with existing code verified
- [ ] Performance impact assessed

Remember: Quality over speed. Each properly completed task builds a solid foundation for the next. Take time to do it right the first time.
</artifact>

These four prompts create a comprehensive workflow that incorporates best practices from the research papers:

1. **Prompt 1** focuses on thorough analysis and documentation, emphasizing accuracy and completeness
2. **Prompt 2** creates a structured research plan that connects findings to specific implementation needs  
3. **Prompt 3** uses both forward and backward reasoning to create a detailed implementation plan
4. **Prompt 4** provides a systematic execution framework with built-in quality controls

Each prompt includes:
- Clear objectives and expected outputs
- Step-by-step guidance
- Self-verification checklists
- Metacognitive prompts to encourage reflection
- Structured output formats for consistency

The prompts are designed to work together as a cohesive system while being comprehensive enough to stand alone if needed.