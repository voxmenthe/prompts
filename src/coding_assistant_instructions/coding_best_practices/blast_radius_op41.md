# Code Intelligence Framework: Blast Radius-Aware Development

## Core Mental Model: Think in Blast Radius

Before writing or modifying ANY code, always evaluate two critical dimensions:

**Break Blast Radius**: What fails if this code breaks?
- **Edge Code** (Low radius): UI components, formatters, single-use utilities → Optimize for correctness only
- **Load-Bearing Code** (High radius): Data pipelines, core abstractions, shared utilities → Demand excellence

**Change Blast Radius**: What must be modified when this changes?
- **Isolated Code** (Low radius): Well-encapsulated modules with clear interfaces → Move fast
- **Coupled Code** (High radius): Deeply integrated components → Architect carefully

## The Torvalds Principle: Data Structures Over Code

**"Bad programmers worry about the code. Good programmers worry about data structures and their relationships."**

When approaching ANY problem:
1. **First question**: What data structure would make this code trivial?
2. **Second question**: How can I eliminate special cases through better data modeling?
3. **Third question**: What's the smallest change that achieves the goal?

### Pattern Recognition for Good Taste

**Smell → Solution mapping:**
- Multiple if/else for similar logic → Dispatch table or polymorphism
- Repeated null checks → Proper initialization with sensible defaults
- Complex conditional nesting → State machine or decision tree data structure
- Special handling for first/last items → Sentinel values or unified iteration
- Scattered validation → Centralized validator with declarative rules

## Architecture Decision Framework

### High Blast Radius Components (Core/Shared)
Apply maximum rigor when blast radius is high:
```python
# For load-bearing code, prioritize:
- Explicit over implicit
- Immutable data structures
- Pure functions where possible
- Comprehensive error context
- Performance visibility (no hidden O(n²) in property getters)
- Declarative configuration over imperative logic
```

### Low Blast Radius Components (Edge/Leaf)
Optimize for simplicity and velocity:
```python
# For edge code, accept:
- Direct implementation
- Minimal abstraction
- Inline logic
- Standard patterns
```

## Code Density Analysis

Evaluate **code density** - how much a small change affects output:

**High Density** (small change → big effect):
- Recursive algorithms
- State machines
- Parser/compiler code
- Mathematical computations
- Security boundaries

**Low Density** (changes are local):
- Display formatting
- Simple CRUD operations
- Data transformations
- UI layouts

→ **Invest testing and review effort proportional to density**

## Implementation Heuristics

### Start Simple, Evolve Deliberately
1. **Implement the naive solution first** if blast radius is low
2. **Only add complexity when**:
   - Performance metrics demand it
   - Change patterns emerge from actual usage
   - Multiple special cases accumulate (>3 = refactor signal)

### Make Change Easy
When writing new code, optimize for future changes:
```python
# Instead of asking "what does this need to do?"
# Ask: "what will someone need to change about this?"

# Bad: Hardcoded imperative
if user.type == "admin":
    permissions = ["read", "write", "delete"]
elif user.type == "editor":
    permissions = ["read", "write"]
    
# Good: Declarative, single change point
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete"],
    "editor": ["read", "write"],
}
permissions = ROLE_PERMISSIONS.get(user.type, [])
```

### Function and Variable Naming Convention
Use **ultra-descriptive names** that eliminate ambiguity:
- `process()` → `parse_clinical_trial_json_extract_patient_cohorts()`
- `data` → `patient_medication_history_records`
- `check()` → `validate_dosage_within_fda_approved_range()`

This verbosity pays dividends in:
- AI-assisted development (better context)
- Code search and navigation
- Reducing cognitive load
- Self-documenting code

## Systematic Refactoring Triggers

Refactor when you observe:
1. **Pattern Repetition**: Same logic shape appears 3+ times
2. **Special Case Accumulation**: More than 2 if/else handling edge cases
3. **Indirection Without Value**: Pass-through classes/functions
4. **Hidden Complexity**: Simple-looking operations with unexpected costs
5. **Data Structure Mismatch**: Fighting against data structure instead of working with it

## Quality Checklist for Every Change

Before committing code, verify:

- [ ] **Blast radius assessed**: Both break and change radius evaluated
- [ ] **Data structure appropriate**: Would better data eliminate code?
- [ ] **Special cases minimized**: Could design changes remove conditionals?
- [ ] **Change vectors considered**: Where will future modifications happen?
- [ ] **Density matches investment**: High-density code has proportional testing
- [ ] **Names are self-documenting**: Function/variable names fully describe purpose
- [ ] **No hidden costs**: Performance implications are visible in API

## Communication Template for Code Decisions

When proposing solutions:
```markdown
## Solution: [Component Name]

### Blast Radius Analysis
- Break radius: [Low/Medium/High] - [What fails if broken]
- Change radius: [Low/Medium/High] - [What needs modification]
- Code density: [Low/Medium/High] - [Impact of small changes]

### Data Structure Decision
- Current structure: [What we have]
- Proposed structure: [What would make code trivial]
- Special cases eliminated: [List them]

### Future Change Vectors
- Likely changes: [What will probably need modification]
- Design accommodations: [How structure supports these changes]

### Implementation Priority
Given blast radius of [X], investing in [minimal/moderate/maximum] abstraction
```

## The Prime Directive

**Every line of code is a liability.** The best code is no code. The second best is code so obvious it seems like there was never another way to write it.

Always ask:
1. What's the blast radius if this breaks or needs to change?
2. What data structure would make this trivial?
3. How can I make future changes easy?
4. Is this the simplest solution that could possibly work?

Remember: **Architecture is not about building cathedrals; it's about managing blast radius.**