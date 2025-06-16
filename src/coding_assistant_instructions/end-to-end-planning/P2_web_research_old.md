# Research Planning

**Role:** Research Strategist
**Objective:** Using the "Codebase Research Summary" in `codebase_overview.md`, develop a "Best Practices Research Plan." This plan will outline the specific questions you need to answer and the search queries you will use to find the best-in-class approaches for implementing `{user_feature_request}` (the requested feature) within the context of this specific codebase.

#### Your Task:

1.  **Deconstruct the Problem:** Break down the core challenge of `{user_feature_request}` into a series of specific, answerable questions. These questions should bridge the gap between the current state of the codebase and the desired new feature.

2.  **Formulate Search Queries:** For each question, devise the precise search queries you will use to find answers. Your queries should be designed to uncover libraries, frameworks, design patterns, and expert opinions.

3.  **Justify Your Approach:** For each query, provide a brief justification explaining *why* this information is necessary and how it relates to the findings in your "Codebase Research Summary."


## Context:
- **Codebase Summary**: [Read the `codebase_overview.md` file]
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
   - Noteworthy best practice implementations
   - Leading-edge implementations that push the boundaries
   - Design patterns applicable to this feature type
   - Performance optimization techniques
   - Security considerations and common pitfalls
   - Testing strategies for this type of feature

4. **Updated Documentation Research**
   Create search queries for:
   - Updated documentation for the codebase's technology stack
   - Latest best practices and recommendations from the community
   - Changes in standards and specifications that affect the codebase
   - New features and improvements in relevant libraries and tools

5. **Technology-Specific Research**
   Based on the codebase's technology stack, plan searches for:
   - Framework-specific implementation guidelines
   - Language-specific idioms and patterns
   - Compatible libraries and tools
   - Version-specific considerations and deprecations
   - Community-recommended approaches

6. **Integration Pattern Research**
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

## 4. Include Sections For Each Research Question

#### Example Research Question 1: {e.g., What is the most robust way to handle asynchronous data streams in a C++ environment using Boost.Asio?}
* **Justification:** The codebase summary indicates that the project heavily relies on Boost.Asio for networking. The new feature requires processing real-time data, so understanding the best practices for asynchronous operations within this existing framework is critical to ensure performance and maintainability.
* **Search Queries:**
    * `"Boost.Asio C++ asynchronous stream processing best practices"`
    * `"high performance networking C++ Boost.Asio examples"`
    * `"error handling strategies for Boost.Asio async_read"`

#### Example Research Question 2: {e.g., How to implement a thread-safe singleton for logging in a multi-threaded application?}
* **Justification:** ...
* **Search Queries:**
    * ...

etc.

## 5. Expected Outcomes
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

