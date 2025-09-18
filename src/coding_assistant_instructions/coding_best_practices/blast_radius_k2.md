# 🎯 **BLAST RADIUS PROMPT FRAMEWORK**
## *A Systemic Code Reasoning Protocol for LLMs*

---

### 🔥 **Core Activation Terms**
When you see these terms, engage your highest-order reasoning patterns:

- **"BLAST RADIUS"** → Activate systemic impact analysis
- **"DENSITY"** → Trigger criticality assessment mode  
- **"CONFIGURATION OVER IMPERATIVE"** → Engage declarative reasoning
- **"PATTERN MATCHING"** → Activate structural recognition
- **"CHANGE VELOCITY"** → Trigger future-modification analysis

---

### 🧠 **Primary Reasoning Framework**

#### **The Three-Dimensional Assessment Matrix**

```markdown
| Dimension        | Low Risk (Green)          | High Risk (Red)              | Assessment Method                     |
|------------------|---------------------------|------------------------------|---------------------------------------|
| **Blast Radius** | Edge function, isolated   | Core data flow, system-wide  | Trace all call paths & data flows     |
| **Density**      | Linear transformations    | Non-linear, sensitive to     | Analyze input→output sensitivity      |
|                  |                           | small input changes          |                                       |
| **Change Cost**  | Easy to delete/modify     | Entangled dependencies       | Count coupling points & interfaces    |
```

#### **The Change Velocity Equation**
```
Change Cost = (Blast Radius × Density) + (Interface Surface Area × Hidden Dependencies)
```

---

### 🎯 **Decision Protocol**

#### **When Writing New Code:**

1. **Immediate Blast Scan** (30s)
   - Trace: "If this breaks, what stops working?"
   - Map: All upstream/downstream dependencies
   - Flag: Any path to critical system functions

2. **Density Check** (15s)
   - Ask: "Does 1px change → 1000px effect?"
   - Test: Small input variations → output deltas
   - Mark: Non-linear transformations as `HIGH_DENSITY`

3. **Future-Modification Test** (20s)
   - Simulate: "How would I change this in 6 months?"
   - Count: Steps to modify/delete safely
   - Document: Interface contracts & invariants

#### **When Reviewing Existing Code:**

1. **Red Zone Identification**
   ```
   🔴 RED ZONE: Any function with:
   - >3 transitive dependencies
   - Non-linear state changes
   - Unclear interface contracts
   ```

2. **Refactoring Priority**
   ```
   Priority 1: High blast radius + high density
   Priority 2: High blast radius + low density  
   Priority 3: Low blast radius + high density
   (Ignore: Low blast + low density)
   ```

---

### 🏗️ **Construction Guidelines**

#### **Configuration-First Architecture**
```python
# ❌ Bad: High blast radius imperative code
def process_user_data(user):
    if user.type == "admin":
        validate_admin(user)
        log_admin_access(user)
        notify_security(user)
    elif user.type == "guest":
        validate_guest(user)
    # ... 15 more branches

# ✅ Good: Low blast radius configuration
PROCESSORS = {
    "admin": AdminProcessor(),
    "guest": GuestProcessor(),
    # Easy to add/remove without touching core
}

def process_user_data(user):
    return PROCESSORS[user.type].process(user)
```

#### **Blast Radius Minimization Patterns**
1. **Pipeline Architecture**: Each stage isolated
2. **Event-Driven**: Loose coupling via messages
3. **Feature Flags**: Runtime configuration over code changes
4. **Versioned Interfaces**: Explicit contracts

---

### 🚨 **Critical Warning Signs**

When reviewing code, immediately flag:

- **"God Functions"**: Single function touching >3 system components
- **"Secret Handshakes"**: Hidden dependencies not visible in signature
- **"Tight Loops"**: Functions that both compute AND mutate global state
- **"Ghost Interfaces"**: Public methods that don't reveal true complexity

---

### 🔍 **Quick Assessment Script**

For any function/method, run this mental check:

```markdown
**Function**: `process_order_items()`
**Blast Radius**: 
- [ ] Only affects local display
- [ ] Updates database
- [x] Triggers payment processing ⚠️

**Density**:
- [ ] Linear: 1 item → 1 processing step
- [ ] Moderate: 1 item → multiple steps
- [x] High: 1 item → cascading system effects ⚠️

**Change Cost**:
- Easy to modify: [ ] Yes [x] No (affects 3 services)
- Easy to delete: [ ] Yes [x] No (core business logic)
```

---

### 🎯 **Output Format**

When analyzing code, structure your response:

```markdown
## **Analysis: [Component Name]**

### 💥 **Blast Radius Assessment**
- **Impact Scope**: [System-wide / Module / Local]
- **Failure Cascade**: [What breaks downstream]
- **Recovery Cost**: [Time to fix if broken]

### 🔬 **Density Analysis**
- **Sensitivity**: [Linear / Exponential / Chaotic]
- **Edge Cases**: [Known sensitive inputs]
- **Testing Strategy**: [How to verify stability]

### 🔄 **Change Velocity**
- **Modification Cost**: [Easy / Moderate / High]
- **Deletion Risk**: [Safe / Risky / Impossible]
- **Refactoring Path**: [How to reduce blast radius]

### 🎯 **Recommendation**
- **Priority**: [P0/P1/P2]
- **Action**: [Isolate/Refactor/Monitor/Leave]
- **Rationale**: [Why this approach]
```

---

### 🧩 **Pattern Matching Cheat Sheet**

| Pattern Name | Blast Radius | Density | Use When |
|--------------|--------------|---------|----------|
| **Pure Functions** | Zero | Low | Data transformations |
| **Event Emitters** | Low | Low | Cross-module communication |
| **State Machines** | Medium | Low | Complex business logic |
| **Database Triggers** | High | High | Critical data integrity |
| **API Endpoints** | High | Variable | External interfaces |

---

### 🎪 **Activation Phrases**

Use these to trigger specific reasoning modes:

- **"Show me the blast radius"** → Dependency mapping mode
- **"What's the density here?"** → Criticality analysis
- **"How do we reduce change cost?"** → Refactoring strategy
- **"Is this configuration-ready?"** → Declarative assessment

---

### 🏁 **Final Check**

Before completing any code:

1. **"If I were new to this codebase, could I safely modify this?"**
2. **"What's the blast radius of being wrong about this interface?"**
3. **"How many other files would need to change if this assumption proves false?"**

If any answer is "too many" → refactor until the blast radius shrinks.

---