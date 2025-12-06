# Integrating `logical-thinking.md` into `.codex/AGENTS.md`

## Objective and Constraints
- Integrate the pre-action logical planning protocol into the agent prompt without inflating blast radius or diluting existing mental models.
- Keep `.codex/AGENTS.md` cohesive (<400 LOC sections) and avoid conflicting directives; prefer declarative, minimal-surface updates.
- Preserve current mission emphases (blast radius, change agility) and avoid redundancy that could cause prompt drift.

## Source Summaries
- `.codex/AGENTS.md`: mission + mental models (blast radius, density, narrow waist), decision protocol (steps 1–6), patterns, naming, coordination/git, quality checklist, prime directive.
- `PLANS/logical-thinking.md`: nine-step meta-reasoning loop focusing on dependencies, risk, hypothesis testing, information completeness, and action inhibition until reasoning is done.

## Integration Options (with blast/change considerations)
### Option A — Inline Meta-Reasoning Step (Lowest friction)
- Add a new "Step 0: Pre-action Logical Scan" atop the decisionProtocol summarizing the nine-step loop in 5–6 bullets.
- Pros: Single file, agents see it in one place; low coordination overhead.
- Cons: Longer decisionProtocol; risk of redundancy with existing blast/density checks; requires careful condensation to avoid verbosity.

### Option B — Layered Annex + Pointer (Minimal prompt growth)
- Add a short pointer in `.codex/AGENTS.md` (e.g., under decisionProtocol intro) that mandates consulting an external annex (kept concise, 8–10 bullets) derived from `logical-thinking.md`.
- Store annex as a separate section at end of `.codex/AGENTS.md` or as an include-style snippet to keep core prompt lean.
- Pros: Limits core prompt size; easy to update annex independently; reduces change blast to main prompt.
- Cons: Reliance on agent following the link; slight risk they skip the annex; two locations to maintain.

### Option C — Execution Loop Overlay (Behavior-first)
- Embed a "Reason → Act → Evaluate" loop graphic in `.codex/AGENTS.md` that explicitly sequences: (1) Dependency/risk scan (steps 1–2 of logical-thinking), (2) Hypothesis planning, (3) Info completeness check, (4) Action gate, (5) Adaptation.
- Pros: Harmonizes both documents into a compact loop; reinforces inhibition before action; fits existing decisionProtocol order.
- Cons: Requires careful wording to avoid over-constraining; moderate prompt edits.

### Option D — Tooling Hooks (Procedural)
- Add micro-checklist near tool usage rubric: "Before any tool call, run dependency/risk/hypothesis triage" with a 3-bullet condensed gate.
- Pros: Targets failure mode (rash tool calls); very small surface; minimal blast radius.
- Cons: Might duplicate Option A/C; risk of scattering instructions.

### Option E — Woven Embedding (Deep integration, no duplication)
- Goal: Meld the nine-step loop into existing `.codex/AGENTS.md` seams so the agent flows through the logic naturally without repeated gates.
- Placement map (one touch per section, concise):
  1) **DecisionProtocol preamble**: add a short preface describing "Step 0: Pre-action Logical Scan" (5 bullets max) as the mandatory front door before Steps 1–6. Do not append gates to each step.
  2) **searchToolsAndParallelism**: prepend a single sentence: "Run Step 0 before any tool call; prefer acting with available info unless blocked by prerequisites."
  3) **coordinationAndGit**: add one line after the list intro: "Pause for Step 0 before commits/rebases to confirm prerequisites, risks, and info completeness."
  4) **qualityChecklist**: add two checklist items: "Step 0 performed (constraints/prereqs cleared)" and "Hypotheses updated after observations."
  5) **workingStyle or ego**: add a sentence on precision/grounding: "Anchor claims in explicit sources/history; avoid assumptions—verify before acting."
  6) **primeDirective footer**: optional one-line reminder to inhibit action until Step 0 is satisfied.
- Wording cues to weave in (drawn from `logical-thinking.md`): mandatory constraints ordering, risk assessment, hypothesis formation, information completeness (tools/history/user), precision/grounding, adaptation after observations, persistence with retries on transient errors.
- Pros: Organic, minimal redundancy; reinforces behavior at natural choke points (before acting, before tools, before commits, at QA gate).
- Cons: Wider touch area raises change blast; needs tight editing to stay concise and avoid token bloat; rollback requires multi-spot edits.

## Recommended Path (balanced blast radius vs. adoption speed)
- Favor Option C (Execution Loop Overlay) for primary integration, with a brief pointer (Option D-lite) near tool usage to reinforce before tool calls.
- Defer full annex (Option B) unless prompt length becomes an issue after drafting; Option A is fallback if a strict single-location policy is preferred.

## Draft Implementation Sketch
1) Insert a new subsection near the top of `decisionProtocol` titled "Step 0: Pre-action Logical Scan" containing a 5-bullet condensation:
   - Identify mandatory constraints/prereqs; reorder tasks if needed.
   - Assess blast/risk of action; prefer acting with available info unless blocked by prereqs.
   - Form top hypotheses; note alternates; plan tests.
   - Confirm info completeness across tools/history/user; avoid assumptions.
   - Inhibit action until above is satisfied; adapt after each observation.
2) Add a short note under `searchToolsAndParallelism` or tool rubric: "Run Step 0 triage before any tool call; defaults to acting with available info when risk is low." 
3) Keep `logical-thinking.md` as the canonical detailed source; optionally link to it from a short "For full rationale" note at the end of the new Step 0.

## Validation Plan
- Dry-run prompts on 3 task types: quick edit, multi-file refactor, exploratory research. Verify agent mirrors Step 0 before tool calls and updates hypotheses after observations.
- Check for regressions: no conflicts with existing blast/density checks; decisionProtocol remains readable.
- If prompt tokens grow too large, switch to Option B (annex) while keeping Step 0 summary <120 words.

## Open Decisions / Dependencies
- Whether to prioritize single-file completeness (Option A/C) vs. lean core plus annex (Option B).
- Token budget limits for `.codex/AGENTS.md` in target runtime.
- Owner for ongoing updates to the annex/loop once integrated.

---

### Option F — Structural Absorption (True organic melding)

**Why Option E falls short**: It still creates a separate "Step 0" that other sections reference. This is fundamentally a bolt-on pattern—the reasoning protocol remains a distinct entity rather than becoming part of the document's DNA. True integration means the nine concepts *disappear into* existing structures.

**Core insight**: The nine elements from `logical-thinking.md` map naturally onto existing `.codex/AGENTS.md` sections. Instead of adding a new gate, we **enrich** what's already there:

| logical-thinking.md concept | Absorbs into |
|----------------------------|--------------|
| 1. Logical dependencies & constraints | `decisionProtocol` Step 1 (Blast Scan) — already traces dependencies; add constraint ordering |
| 2. Risk assessment | `decisionProtocol` Step 2 (Density Check) + new `architectureDecisionHeuristics` item |
| 3. Abductive reasoning / hypothesis | `errorHandlingObservabilityFallbacks` + `patternRecognitionHeuristics` — already about root cause |
| 4. Outcome evaluation / adaptability | `decisionProtocol` preamble — add adaptation mandate |
| 5. Information availability | `searchToolsAndParallelism` intro — already about using tools; enrich with explicit sources list |
| 6. Precision & grounding | `workingStyle` (if exists) or new `groundingPrinciples` subsection in mission area |
| 7. Completeness | `qualityChecklist` — add verification item |
| 8. Persistence & patience | `errorHandlingObservabilityFallbacks` — add retry/persistence policy |
| 9. Inhibit action | `decisionProtocol` outro/footer — single sentence; ties to prime directive |

**Implementation map** (minimal touches, maximum absorption):

1. **Mission statement** (line ~4): Append one sentence linking adaptive reasoning to deliberate action:
   > "Before any action, systematically reason through dependencies, risks, and information completeness."

2. **New mental model** `deliberate-action`: Add to `coreMentalModels` a model that captures the *why* of pre-action reasoning, framed as architectural principle (fits the existing style). ~6 lines.

3. **DecisionProtocol preamble** (before Step 1): Add 2-3 sentences establishing the protocol is iterative and adaptive—plans update after observations, hypotheses evolve. No "Step 0"; just context-setting.

4. **Step 1 (Immediate Blast Scan)**: Enhance with constraint ordering: "Resolve conflicts by: policy rules → order of operations → prerequisites → user preferences."

5. **Step 2 (Density Check)**: Add risk framing: "Assess consequences; for exploratory/low-risk actions, prefer acting with available info over blocking for optional parameters."

6. **errorHandlingObservabilityFallbacks**: Add 2 items:
   - Hypothesis-driven debugging: "Form explicit hypotheses before debugging; test them systematically; don't discard low-probability causes prematurely."
   - Persistence policy: "On transient errors, retry unless explicit limit is hit. On other errors, change strategy rather than repeating failed approach."

7. **searchToolsAndParallelism**: Prepend to existing intro:
   > "Incorporate all information sources before acting: tool outputs, policies/constraints, conversation history, and (only when necessary) user clarification."

8. **qualityChecklist**: Add one item: "Claims grounded in explicit sources; no unverified assumptions acted upon."

9. **primeDirective** (line ~367): Append one sentence (the "inhibition" concept, reframed positively):
   > "Reason fully before acting; once committed, you cannot uncommit."

**Why this is better than E**:
- **No new structural element** (no "Step 0" to reference)
- **Concepts become invisible** — they're woven into existing language
- **Smaller token footprint** — enriches ~8 existing locations with 1-3 sentences each instead of creating a new 5-bullet section plus 6 cross-references
- **Lower change blast** — each edit is localized and doesn't create dependencies on a new concept name
- **No redundancy** — maps each logical-thinking concept to exactly one home

**Rollback story**: Because changes are small enrichments to existing sections, reverting any individual enhancement is trivial and doesn't break other sections.

**Estimated token delta**: +150–200 tokens (vs. Option E's ~250–300 due to the Step 0 block plus repeated references).

---

## Revised Recommendation

**Adopt Option F** for primary integration. It achieves the goal of "deep organic melding" by distributing the nine reasoning principles into their natural homes within the existing structure. The agent will exhibit the desired pre-action reasoning behavior without ever seeing a distinct "Step 0" gate.

If future prompt-length constraints emerge, selectively trim the enrichments (they're independent) rather than extracting to an annex.
