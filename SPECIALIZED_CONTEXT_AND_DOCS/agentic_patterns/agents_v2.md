**How I’ll tackle this:** I’ll reframe “Deep Agents (Agent 2.0)” as a software architecture problem rather than a prompt trick. I’ll (1) delineate the shortcomings of shallow loops in precise engineering terms, (2) specify a framework‑agnostic architecture with clean interfaces, (3) show semi‑detailed, real‑world examples that map design choices to failure modes, and (4) include high‑level pseudocode for the core primitives (planning, delegation, memory, context composition, and recovery). The goal is something you can hand to an engineering team and implement without committing to any particular library.

---

# Deep Agents, Done Right: A Framework‑Agnostic Playbook

## 0) Quick recap: why “shallow loops” fail in production

A classic Agent 1.0 looks like this:

```
while not done:
  thought = LLM(prompt + recent_messages)
  if tool_call in thought:
      result = call_tool(tool_call)
      append(result)
  else:
      return finalize(thought)
```

This is simple and powerful for short tasks, but it breaks down for anything long‑horizon:

* **State starvation.** The *only* state is the context window. Tool dumps push out instructions and goals. You see thrash, regressions, or forgotten constraints.
* **No global plan.** Decisions are myopic. The agent can’t reason about dependency order, budgets, or stopping conditions, so it ping‑pongs between tools.
* **No durable progress.** If the process restarts or the window is exceeded, the “project memory” evaporates. You can’t run for hours or days safely.
* **No isolation.** One messy subtask contaminates the whole context. Errors fan out, observability is poor, and failures aren’t scoped.

## 1) Agent 2.0 in one sentence

**A Deep Agent is a stateful, planned, hierarchical system with durable memory and protocol‑rich prompts.** It separates *plan* from *act*, isolates work into sub‑agents with minimal privileges, writes artifacts to persistent stores, and composes context deterministically.

Think “workflow engine with LLM operators,” not “while‑loop that sometimes calls tools.”

---

## 2) Architectural pillars (with implementation primitives)

### Pillar A — Explicit Planning

**What it is.** Convert a goal into a structured plan (DAG or ordered list), track status transitions (`PENDING → RUNNING → DONE/FAILED`), and replan when reality diverges.

**Plan representation (minimal):**

```yaml
Plan:
  goal: string
  acceptance: list[string]           # measurable criteria
  budget: {tokens:int, time:sec, cost:currency}
  steps:
    - id: "S1"
      title: "Collect competitor pricing"
      role: "Researcher"
      depends_on: []
      inputs: {}
      outputs: {artifact: null}
      status: "PENDING"
      retries: 0
    - id: "S2"
      title: "Model price sensitivity"
      role: "Analyst"
      depends_on: ["S1"]
      inputs: {artifact_ref: "S1.artifact"}
      outputs: {model: null}
      status: "PENDING"
```

**Planner loop (pseudocode):**

```pseudo
function plan(goal, constraints) -> Plan:
  steps = decompose_with_llm(goal, constraints)
  return normalize(steps)            # add ids, deps, budgets, acceptance

function schedule(plan) -> list[Step]:
  ready = [s for s in plan.steps if deps_done(s)]
  return prioritize(ready, by=critical_path_and_budget)

function replan(plan, telemetry):
  if violated(telemetry.acceptance or telemetry.budget or telemetry.deadline):
      deltas = llm_propose_changes(plan, telemetry)
      return apply_deltas(plan, deltas)
  return plan
```

**Engineering notes**

* *Idempotency:* Make every step idempotent (e.g., dedupe by `(plan_id, step_id, attempt)`).
* *Determinism:* Keep planning prompts deterministic by pinning schemas and examples; store the planner’s decisions as events.

---

### Pillar B — Hierarchical Delegation (Orchestrator ↔ Sub‑Agents)

**What it is.** An **Orchestrator** owns the plan and the budgets. It spawns **Sub‑Agents** (specialists) with *scoped prompts, tools, and context budgets*. Sub‑agents return artifacts—not their entire internal scratchpad.

**Process boundary:** Run sub‑agents in **ephemeral contexts** (fresh message history, dedicated tool set). Feed them only what they need: the step spec, relevant memory, and acceptance criteria.

**Sub‑agent contract (high‑level):**

```json
{
  "step_id": "S1",
  "role": "Researcher",
  "goal": "Collect competitor pricing across top 10 vendors.",
  "constraints": {
    "sources": ["public web", "recent PRs"],
    "budget": {"tokens": 8000, "time_sec": 600}
  },
  "inputs": {},
  "deliverable_schema": {
    "type": "table",
    "columns": ["vendor","product","price","source_url","confidence"]
  }
}
```

**Orchestrator loop with delegation (pseudocode):**

```pseudo
function run_plan(plan_id):
  plan = load_plan(plan_id)
  while not all_done(plan):
    for step in schedule(plan):
      step.status = RUNNING; save(plan)
      try:
        ctx = compose_context(step)               # see Pillar D
        result = run_subagent(step.role, ctx)     # isolated
        ref = persist_artifact(plan.id, step.id, result)
        complete(step, outputs={artifact: ref})
      catch e:
        mark_failed(step, error=e)
        if step.retries < MAX:
          step.retries += 1; enqueue(step)
        else:
          escalate_or_replan(plan, step)
    plan = replan(plan, collect_telemetry(plan))
    checkpoint(plan)
  return finalize(plan)
```

**Isolation rules**

* Least privilege: only the tools needed for the role.
* Time/step budget per sub‑agent.
* Output **must** match a declared schema; reject on validation failure.

---

### Pillar C — Persistent Memory (Artifacts + Event Log + Retrieval)

**What it is.** Treat memory as a **first‑class data layer**:

* **Artifact store:** versioned files/tables/code, addressable by URIs (`artifacts://{plan}/{step}/{hash}`).
* **Event log:** append‑only record of decisions (`plan.created`, `step.started`, `artifact.written`…).
* **Retrieval index:** embeddings + metadata (source, timestamp, owner, tags) with policies to avoid junk.

**Write path:**

```pseudo
function persist_artifact(plan_id, step_id, content):
  uri = write_blob(content)              # immutable
  index(uri, embeddings(content), meta={plan_id, step_id, ts})
  log("artifact.written", {plan_id, step_id, uri})
  return uri
```

**Read path (policy‑driven retrieval):**

```pseudo
function recall(query, filters) -> list[Snippet]:
  candidates = vector_search(query, k=50, filters)
  ranked = rerank(query, candidates)     # keyword + metadata + recency
  window = budgeted_select(ranked, token_budget=K)
  return summarize_if_needed(window)     # shrink to fit
```

**Memory hygiene**

* Store sources and lineage with each artifact (who/what produced it).
* TTL or compaction policies for intermediate debris.
* “Golden” summaries: small, manually reviewed snapshots of critical knowledge.

---

### Pillar D — Extreme Context Engineering (Protocols, not vibes)

**What it is.** Prompts are **interfaces**. Encode roles, tool specs, message formats, error‑handling, and stop conditions. Keep them versioned and testable.

**Execution envelope (what the model sees):**

```
SYSTEM:
  - Role: Researcher
  - Tools: {web.search, web.fetch, csv.write}
  - Protocols:
      * Always plan before acting: output PLAN{} then TASKS[].
      * Use JSON blocks for tool calls: {"tool": "...", "args": {...}} only.
      * Deliverable schema: Table(vendor, product, price, source_url, confidence).
      * Stop when ACCEPTANCE passes or budget exhausted.
  - Safety: Never access credentials or internal URLs.

CONTEXT:
  - Plan excerpt: step S1 spec...
  - Relevant memory snippets (<= 2k tokens)...
  - Examples (1-2) of correct deliverable format...

USER:
  "Collect competitor pricing across top 10 vendors."
```

**Validator (outside the model):**

```pseudo
function enforce_contract(model_output):
  blocks = extract_json_blocks(model_output)
  for b in blocks:
    if b.tool: validate_toolcall_schema(b)
  deliverable = extract_deliverable(model_output)
  assert conforms(deliverable, deliverable_schema)
```

---

## 3) Cross‑cutting patterns that make or break reliability

* **Durability:** checkpoint the plan + event log frequently. On restart, rebuild in‑memory state from the log, not from chat history.
* **Idempotency + de‑dup:** step outputs are content‑addressed (hash). Replays don’t duplicate work.
* **Budgets:** explicit token/time/cost budgets at plan and step levels; decremented by runtime telemetry.
* **Circuit breakers:** backoff and open the circuit on repeated tool failures or hallucinated tool names.
* **Observability:** trace ids for plan/step/attempt; keep prompts, responses, tool args, validation results.
* **Human‑in‑the‑loop:** transitions `AWAITING_APPROVAL` and “explain why” summaries to reduce review time.
* **Policy enforcement:** allow/deny lists for tools and domains; redactors for PII before indexing memory.

---

## 4) Semi‑detailed real‑world implementations

### Example 1 — Multi‑day RFP → Proposal Autopilot

**Goal.** Read an RFP package, extract requirements, map capabilities, draft a custom proposal, and produce a pricing appendix over 72 hours with periodic human checks.

**Decomposition.**

* S1 *Ingest & normalize RFP*: extract sections, deadlines, submission format.
* S2 *Requirements matrix*: tabulate must/should/could with IDs.
* S3 *Capability mapping*: map each requirement to internal capability docs.
* S4 *Gap analysis*: flag unmet items + suggested mitigations.
* S5 *Proposal draft*: assemble narrative + tables.
* S6 *Pricing appendix*: compute BOM from catalog, apply discount policy.
* S7 *Compliance checklist*: verify page limits, fonts, attachments.

**Sub‑agents.**

* `DocIngestor` (PDF/Doc parsers, OCR).
* `Indexer` (embeddings, metadata enrichment).
* `Mapper` (RAG over internal corpus).
* `Writer` (narrative with templates).
* `Pricer` (policy‑constrained spreadsheet ops).
* `QA` (format/policy validators).

**Key robustness tricks**

* “Golden requirements table” is a **single source of truth**; every later step references it by ID.
* The `Mapper` returns *artifact references* (document URIs + spans) not raw text, enabling traceability.
* `Writer` uses a **templated scaffold** with placeholders bound to artifacts; validators ensure no unbound placeholder remains.

**Pseudocode slice (S3 Mapper):**

```pseudo
function map_requirements(req_table_uri):
  reqs = read_table(req_table_uri)
  results = []
  for req in reqs:
    query = f"{req.id}: {req.text}"
    snippets = recall(query, filters={source:"internal"})
    evidence = select_top_k(snippets, k=5)
    results.append({
      "req_id": req.id,
      "evidence_refs": [s.uri for s in evidence],
      "coverage_summary": llm_summarize(evidence),
      "confidence": calibrate(evidence)
    })
  return persist_artifact(plan.id, "S3", to_table(results))
```

**Human gates**

* After S4 (gaps), require approval before drafting (S5).
* After S6 (pricing), require finance approval.

---

### Example 2 — SRE Runbook Autopilot (with guardrails)

**Goal.** During an incident, triage alerts, gather metrics/logs, propose a fix, and craft a status update. The agent must never execute destructive actions.

**Decomposition.**

* S1 *Classify incident* (service, severity, hypothesis).
* S2 *Evidence gathering* (logs, metrics; read‑only).
* S3 *Diff against known issues*.
* S4 *Candidate runbook selection*.
* S5 *Dry‑run fix plan*, list exact commands and expected outcomes.
* S6 *Comms drafts* (Slack/StatusPage template).
* S7 *Handover & escalation* (assign to human).

**Safety**

* Read‑only access only; no shell execution. Actions are **proposals**.
* Tooling returns **structured datasets** (tables of metrics, log excerpts).
* Validators enforce: every step must cite evidence (URIs) for claims.

**Pseudocode slice (S2 evidence gathering):**

```pseudo
function gather(service, timeframe):
  logs = call("observability.logs.query", {service, timeframe, limit:2000})
  metrics = call("observability.metrics.query", {service, timeframe, series:[
    "latency_p95","error_rate","throughput"
  ]})
  return persist_artifact(plan.id, "S2", {"logs": logs.uri, "metrics": metrics.uri})
```

**Context composition for S5 (fix plan):**

```
SYSTEM: Role=FixPlanner, Tools=[], Protocols=produce PLAN then COMMANDS[]
CONTEXT:
  - Incident: {service, severity, hypothesis}
  - Evidence: summaries of metrics/logs with URIs
  - Allowed actions: "proposal only"; no execution permitted
USER: "Produce a step-by-step remediation plan with rationale and rollback."
```

**Outcome**

* Human on‑call gets a machine‑readable plan and can click “approve & execute” in a separate system that enforces RBAC and dry‑run diffs.

---

### Example 3 — Quarterly FP&A “Pack” Builder

**Goal.** Build a quarterly narrative: ingest earnings releases, call transcripts, and internal actuals; produce a variance analysis and a board‑ready deck.

**Decomposition.**

* S1 *Collect externals* (press releases, transcripts).
* S2 *Ingest internal actuals* (CSV/warehouse queries)
* S3 *Variance analysis* vs forecast.
* S4 *Narrative drafting* (key drivers, risks, outlook).
* S5 *Charts* (deterministic code to render).
* S6 *Deck assembly* (bind artifacts into slides).

**Determinism and testing**

* Chart generation is **non‑LLM** (template code). The LLM only produces chart specs, which go through a schema validator.
* Unit tests over the variance computation; LLM is not in the loop for numbers.

**Chart spec contract:**

```json
{
  "chart_id": "gm_margin_trend",
  "type": "line",
  "data_source": "S3.variance_table",
  "x": "quarter",
  "y": ["gm_actual","gm_forecast"],
  "title": "Gross Margin Trend",
  "notes": "Highlight Q2 anomaly due to COGS spike."
}
```

**Pseudocode slice (chart render):**

```pseudo
def render_chart(spec):
  assert schema_ok(spec)
  df = load(spec.data_source)
  fig = plot(df, type=spec.type, x=spec.x, y=spec.y, title=spec.title)
  return persist_artifact(plan.id, "S5", save_png(fig))
```

---

### Example 4 — Contract Intake & Redline Assistant

**Goal.** Triage inbound contracts, classify risk, propose redlines, and produce a negotiation brief.

**Decomposition.**

* S1 *Extract clauses* into normalized structure (party, term, liability, ip, etc.).
* S2 *Policy compare* vs playbook (thresholds per clause).
* S3 *Risk scoring* with explanations and precedents.
* S4 *Redline suggestions* (inline diffs).
* S5 *Negotiation brief* (talking points + fallback positions).

**Memory design**

* Maintain a **precedent library** of accepted/denied clause variants, tagged by counterparty and date; retrieval is by clause type + risk dimension.
* “Golden redlines” approved by legal become exemplars; future runs prefer them.

**Redline generation (diff‑first strategy):**

```pseudo
function propose_redline(contract_clause, playbook):
  target = playbook.preferred_text[contract_clause.type]
  diff = compute_diff(contract_clause.text, target)      # algorithmic
  prompt = build_prompt("RedlineWriter",
                        context={clause: contract_clause.text, target, diff})
  out = llm(prompt)
  return validate_and_structure(out, schema="redline_patch")
```

**Guardrail**

* Redline patches are **patch objects** (unified diff) applied by a safe differ, not arbitrary text replacements.

---

## 5) High‑level system pseudocode (all pillars together)

```pseudo
# Types
type Step   = {id, title, role, depends_on[], inputs{}, outputs{}, status, retries}
type Plan   = {id, goal, acceptance[], budget, steps[], created_at, version}
type Event  = {ts, plan_id, step_id?, type, payload}
type ArtifactRef = {uri, sha256, meta{}}

# Entry point
function execute(goal, constraints) -> Plan:
  plan = plan(goal, constraints)
  log_event("plan.created", {plan})
  checkpoint(plan)

  while not all_done(plan):
    for step in schedule(plan):
      if budget_exceeded(plan): break
      start_step(step)

      try:
        ctx = compose_context(plan, step)               # Pillar D
        result = run_subagent(step.role, ctx)           # Pillar B
        ref = persist_artifact(plan.id, step.id, result) # Pillar C
        finish_step(step, outputs={artifact: ref})
      catch err:
        handle_failure(plan, step, err)
      finally:
        checkpoint(plan)

    plan = replan(plan, collect_telemetry(plan))        # Pillar A

  return plan

# Failure handling
function handle_failure(plan, step, err):
  log_event("step.failed", {step.id, err})
  if is_transient(err) and step.retries < MAX_RETRIES:
     step.retries += 1; requeue(step, after=backoff(step.retries))
  else:
     mark_failed(step)
     if critical(step): escalate(plan, step)
     plan = replan(plan, {reason:"failure", step_id:step.id})
```

**Context composer (budget‑aware retrieval + contracts):**

```pseudo
function compose_context(plan, step):
  # 1) Always include step spec + acceptance + budgets
  base = [format_step_spec(step)]
  # 2) Retrieve only relevant memory (strict filters)
  mem = recall(step.title + " " + plan.goal,
               filters={plan_id: plan.id, role: step.role})
  # 3) Include 1‑2 good exemplars
  examples = sample_examples(step.role, step.title)
  # 4) Budget and tokenize
  return budgeted_pack(system=protocol_for(step.role),
                       context=base + mem + examples,
                       token_limit=ROLE_LIMITS[step.role])
```

---

## 6) Testing, evaluation, and operations

* **Golden flows:** Save full traces (inputs, prompts, outputs, artifacts) for representative runs. Re‑run after any prompt or tool change.
* **Property tests:** Validate invariants (e.g., “every claim cites a source URI”, “proposal length ≤ page limit”, “numbers sum”).
* **Canaries:** Shadow‑run the agent on live data with outputs gated; measure *plan progress*, *retries*, and *error taxonomy*.
* **Budget accounting:** Attribute tokens/time/cost by step; use this to prune waste (e.g., too much context in S3).
* **Prompt versioning:** Prompts and schemas are versioned assets; include `prompt_version` in every event.
* **Observability dashboards:** Live views for plan DAG status, artifact lineage, hot errors, and budget consumption.

---

## 7) Practical checklist (copy/paste)

1. **Define contracts first.** Step schemas, deliverable schemas, tool I/O. No free‑form outputs across boundaries.
2. **Make planning explicit.** Plan artifact is the source of truth; replan on budget drift, failures, or new info.
3. **Isolate roles.** New sub‑agent when domain, toolset, or risk profile changes. Least privilege always.
4. **Write everything down.** Artifacts immutable; events append‑only; every result has lineage.
5. **Compose context deterministically.** Retrieval filters + budgets + examples. No “dump the world”.
6. **Validate every boundary.** JSON schema, length limits, citation checks, unit tests for non‑LLM steps.
7. **Instrument.** Trace IDs, prompts, tool calls, retries. You can’t debug what you can’t see.
8. **Guardrails & humans.** Circuit breakers, escalation states, and cheap summaries for reviewers.

---

## 8) What Deep Agents unlock—and what still needs work

**Unlocked today**

* Multi‑hour/day workflows with true *durability* (restart mid‑plan without losing the thread).
* Modular teams of agents that *specialize* without stepping on each other’s context.
* Transparent, auditable pipelines where artifacts and decisions have provenance.

**Still hard**

* **Inter‑agent negotiation.** Beyond “supervisor dispatch,” we need principled consensus/critique patterns with guarantees on convergence.
* **Memory routing quality.** The last 20% reliability is about retrieving *the* right slice at *the* right time, under strict budgets.
* **Cost/latency governance.** Budget‑aware planning that automatically trades model size, context length, and tool depth against deadlines.

---

### Closing thought

Treat your Agent 2.0 like any distributed system: explicit state, durable logs, strict interfaces, isolation, and observability. Use LLMs as *operators* inside that system—not as the system. When you do, the leap from “clever chatbot” to **reliable, long‑horizon automation** stops being aspirational and starts being an engineering exercise you can ship.


Below are **three new sections**—drop‑in additions to the playbook—that go deeper on the topics you called out. They are framework‑agnostic and designed to plug into the architecture and pseudocode you already have.

---

## 9) Memory routing quality — retrieving the right slice at the right time

**Goal.** Deliver only *decision‑critical* context to the model, within tight token/time budgets, while preserving provenance and avoiding retrieval spam.

### 9.1 Principles

1. **Many small, typed memories, not one big blob.** Partition memory into *lanes* (e.g., `decisions`, `golden_summaries`, `artifacts`, `logs`, `examples`) with explicit schemas and TTLs. Retrieval is lane‑aware.
2. **Queries are programs.** Build queries from step spec + acceptance criteria + dependencies (not just the step title). Include *must* and *must‑not* filters.
3. **Stage retrieval, then compress.** (a) direct references → (b) strict metadata filters → (c) lexical retrieval → (d) embedding retrieval → (e) re‑rank → (f) budgeted packing with compression.
4. **Evidence or it didn’t happen.** Every snippet included must carry a URI + span + timestamp + producer. Drop orphaned text.
5. **Budget is a first‑class input.** The router plans tokens across lanes before fetching. Treat packing as a knapsack problem with utility weights.
6. **Feedback improves routing.** Capture *utility signals* (did the snippet get cited? did the validator accept it? was it contradicted?) and learn lane weights over time.

### 9.2 Routing pipeline (pseudocode)

```pseudo
# Lanes: "decisions", "golden", "artifacts", "examples", "logs"
# Utility weights reflect prior usefulness; can be learned online.

function route_memory(plan, step, token_budget K) -> PackedContext:
  Q = build_query(step, plan)  # goal, acceptance, inputs, deps, entities, date range

  # Stage 0: direct refs from step inputs/deps (no search)
  S0 = materialize_direct_refs(step.inputs, step.depends_on)

  # Stage 1: strict metadata filters
  F = {
    plan_id: plan.id,
    role: step.role,
    recency_days: window_for(step),
    verified: true
  }
  S1 = search_metadata(F)  # small

  # Stage 2: lexical/BM25 (high precision for exact terms)
  S2 = bm25(Q, topk=100, lanes=["decisions","golden","artifacts"])

  # Stage 3: dense retrieval (semantic)
  S3 = embed_and_search(Q, topk=300, lanes=["artifacts","logs"])

  # Stage 4: cross-lane re-rank with MMR + recency + authority
  C = combine(S0,S1,S2,S3)
  R = rerank(C, key = lambda s: alpha*text_sim(Q,s)
                         + beta*recency(s)
                         + gamma*authority(s.source)
                         - delta*redundancy(s))

  # Stage 5: budgeted packing with lane quotas and on-the-fly compression
  quotas = allocate_quota(K, weights={"decisions":3, "golden":2, "artifacts":2, "examples":1, "logs":0.5})
  packed = []
  for lane in lane_order(quotas):
    for s in R where s.lane==lane:
      if fits(packed, s, quotas[lane]): packed.append(s)
      else if compressible(s): packed.append(summarize_to_fit(s))
      if tokens(packed) >= K: break

  # Stage 6: final provenance & schema validation
  packed = [s for s in packed if has_provenance(s) and schema_ok(s)]
  return build_context_blocks(packed)
```

**Notes**

* `window_for(step)` tightens recency for time‑sensitive steps.
* `authority()` is a source prior (e.g., curated corpus > raw web).
* `redundancy()` uses approximate submodularity (MMR) to favor diversity.
* `summarize_to_fit()` uses a fixed schema (bullet summary + citations), capped at N tokens.

### 9.3 Query builder (deterministic)

```pseudo
function build_query(step, plan):
  terms = [plan.goal, step.title] + plan.acceptance
  entities = extract_entities(step.inputs) + extract_entities_from_deps(step.depends_on)
  neg = ["deprecated", "superseded"]  # lane-specific must-not tags
  return {
    must: dedupe(terms + entities),
    must_not: neg,
    filters: {plan_id: plan.id, role: step.role}
  }
```

### 9.4 Budget‑aware packing (knapsack view)

Treat each candidate snippet `i` with value `v_i` and cost `c_i` (estimated tokens). Solve greedily with re‑ranking as heuristic (it’s near‑optimal in practice), or exact if K is small. Integrate *lane quotas* to preserve a minimum allocation to high‑value lanes like `decisions` and `golden`.

### 9.5 Practical examples

**A) RFP autopilot (S3: capability mapping).**

* Lane priorities: `golden` (past accepted answers for similar requirements), `artifacts` (internal capability docs), `decisions` (prior mapping choices), `examples` (good mapping format).
* Strict filter: `requirement_id IN current_batch` to avoid flooding with unrelated artifacts.
* Compression: convert long capability docs to *evidence bullets* with citation URIs.

**B) SRE runbook (S2: evidence gathering).**

* Lane priorities: `artifacts` (metrics tables), `logs` (aggregated error samples), `decisions` (previous incidents same service), `golden` (known issues).
* Token cap K=1200; allocate 40% to metrics summary, 40% to known issues, 20% to logs excerpts.
* Drop any log snippet without a regex‑matched incident ID.

**C) FP&A (S4: narrative drafting).**

* Pull `golden` quarterly “driver summaries” first, then `artifacts` variance tables.
* Enforce “cite or drop”: every narrative claim must reference an artifact URI; non‑cited context is excluded from the pack.

### 9.6 Quality control & learning

* **Offline eval:** maintain gold query→needed‑snippets sets; track precision@K, coverage@acceptance, and *cited@K*.
* **Online signals:** promote snippets that get cited and pass validator; demote never‑used sources; blacklist low‑precision domains.
* **Drift control:** auto‑regenerate `golden_summaries` from frequently used artifacts; expire rarely retrieved items (TTL).
* **Fail‑safe:** if packing can’t satisfy acceptance‑critical needs (e.g., no `decisions` lane included), trigger a *memory repair step* to synthesize a compact “brief” artifact and retry routing.

---

## 10) Inter‑agent negotiation — principled consensus/critique with convergence

**Goal.** Obtain better decisions than any single agent by structured disagreement, with bounded rounds and clear termination.

### 10.1 Roles & artifacts

* **Proposer(s):** produce candidate plans/answers with citations.
* **Critic(s):** enumerate concrete objections (errors, risks, missing evidence).
* **Judge:** scores candidates against a rubric and the objections; decides accept/revise/reject.
* **Ledger:** an artifact that stores Claims, Evidence, Objections, and Decisions (with URIs).

**All messages use schemas.** No free‑form debate across boundaries.

### 10.2 Rubric (example)

```yaml
rubric:
  objectives_weight: 0.35     # meets acceptance criteria
  evidence_weight:   0.30     # citations, coverage, recency
  risk_weight:       0.20     # safety, reversibility, blast radius
  cost_weight:       0.15     # within budget/time
  thresholds:
    min_total: 0.72
    disqualifiers:
      - "no citations"
      - "violates safety policy"
```

### 10.3 Negotiation protocol (bounded)

```pseudo
function negotiate(task, max_rounds=3):
  ledger = new_ledger(task_id=task.id)

  # Round 0: diverse proposals (n>=2) to avoid premature convergence
  C = [Proposer_i.propose(task) for i in 1..N]
  record(ledger, "proposals", C)

  best = None; best_score = -inf

  for r in 1..max_rounds:
    # Critics attack each candidate with structured objections
    for c in C:
      objections = [Critic_j.critique(c) for j in 1..M]
      record(ledger, "objections", {c.id: objections})

    # Proposers revise addressing objections (must link resolutions)
    C = [Proposer_k.revise(c, ledger.objections[c.id]) for c in C]
    record(ledger, "revisions", C)

    # Judge scores with rubric; keep top-K
    scored = [(c, Judge.score(c, rubric)) for c in C]
    best_new, score_new = argmax(scored)
    record(ledger, "scores", scored)

    # Convergence tests
    if score_new >= rubric.thresholds.min_total: best, best_score = best_new, score_new; break
    if stable(ledger) or not improves(score_new, best_score, eps=0.01): break
    best, best_score = best_new, score_new
    C = topk(scored, K=2)  # prune to strongest to save budget

  if best_score < rubric.thresholds.min_total:
    escalate_to_human(ledger.summarize())
  else:
    commit_decision(ledger, best)
  return best, ledger
```

**Convergence guarantees**

* **Bounded rounds** ⇒ termination in ≤ `max_rounds`.
* **Monotone best score** ⇒ with pruning and a fixed rubric, the best‑so‑far is non‑decreasing; if it stops improving (`eps`), halt.
* **No‑new‑arguments test** ⇒ hash the set of unresolved objections; if unchanged across a round, halt.

### 10.4 Practical patterns

**A) SRE remediation plan**

* 2 Proposers: *Primary* (speed‑biased) and *Safety* (risk‑averse).
* Critics: *BlastRadius*, *Rollback*, *Compliance*.
* Judge weights risk higher at Sev‑1.
* Termination if either plan reaches `min_total` or time budget exceeded; otherwise auto‑escalate with the highest‑scoring plan + objections summary.

**B) Contract redlines**

* Proposers generate patch objects.
* Critics check clause‑policy violations, cite precedents.
* Judge validates patches apply cleanly and meet negotiation targets.
* Two‑phase commit: Legal and Sales agents must both *prepare* (ack) → then *commit*. If either rejects, reopen negotiation with constraints tightened.

**C) Data pipeline schema change**

* Proposer suggests migration plan (backfills, dual‑write).
* Critics: *Cost*, *DataQuality*, *Downtime*.
* Judge requires zero‑data‑loss proof and rollback time ≤ X minutes.
* Decision recorded in ledger; CRDT merge for config artifacts prevents lost updates.

**Operational tips**

* Keep each role’s prompt short and protocol‑heavy; rotate “roles” across the same underlying model to avoid bias.
* Weight votes by **historical reliability** on similar tasks.
* Maintain a **dissent log**: even if a plan is accepted, persist unresolved objections for post‑mortem.

---

## 11) Cost/latency governance — budget‑aware planning and adaptive compute

**Goal.** Meet deadlines and cost caps by dynamically trading model size, context length, and tool depth—without tanking quality.

### 11.1 Compute model & SLOs

Define a simple cost/latency model per resource:

```
Model m ∈ {S, M, L}:
  cost(m)      = a_prompt*prompt_tokens + a_comp*gen_tokens
  throughput(m)= tokens/sec
  overhead(m)  = fixed_latency

Tool t:
  cost(t)      = api_cost + compute_cost
  latency(t)   = measured_ms
```

Set SLOs per plan/step: `{deadline_sec, max_cost, quality_target}`.

### 11.2 Gating policy (coarse‑to‑fine)

1. **Default small model** for planning, retrieval, and simple transforms.
2. **Escalate** to larger models only when *signals* warrant it: low confidence, validator failures, high impact, or ambiguity.
3. **Constrain context** before escalating: compress memory, reduce examples, prefer `golden` lanes.
4. **Early exit** when acceptance tests pass.

**Signals (examples)**

* Validator failure count ≥ N
* Self‑reported uncertainty ≥ τ (bounded scale)
* Disagreement among agents (score gap small)
* Impact score high (e.g., Sev‑1, board deck)
* Time remaining >> estimated remaining work at higher model size

### 11.3 Budget‑aware scheduler (pseudocode)

```pseudo
function schedule_budget(plan):
  B = plan.budget  # {tokens, time, cost}
  Q = critical_path(plan)  # step weights by impact/dependency

  for step in topological_order(plan):
    est = estimate(step, model="S")    # tokens, time, cost using telemetry
    if violates(est, B):                 # not feasible with S
      m = choose_model(step, B, est)     # S→M→L as needed
      ctx = tighten_context(step)        # reduce lanes/examples first
    else:
      m = "S"
    assign(step, model=m, ctx=ctx)
    reserve(B, est_for(m, step))
```

**Model chooser**

```pseudo
function choose_model(step, B, est_S):
  for m in ["M","L"]:
    est = estimate(step, model=m)
    value = expected_quality_gain(step, m)   # from telemetry or heuristics
    if est.within(B) and value/est.cost >= VOI_threshold:
      return m
  return "S"  # fallback; will likely trigger human gate later
```

### 11.4 Tool depth & recursion caps

Set per-role limits: `max_tool_calls`, `max_depth`, `max_parallelism`. When exceeded, force a *summarize & replan* action rather than blindly continuing. For web tasks, enforce a crawl budget and restrict domains.

### 11.5 Practical strategies

**A) RFP autopilot under 2‑hour deadline**

* Plan time budget across steps using critical path; allocate 60% to mapping & drafting.
* Use Small model for S1/S2 ingestion; Medium for S3 mapping; Large only for S5 narrative *if* rubric score < threshold on first pass.
* Aggressive context tightening: allow only `decisions` and `golden` lanes for the Large call.

**B) SRE incident at Sev‑1 (TTR ≤ 15 minutes)**

* All steps default to Small/Medium; Large is disabled (latency risk).
* Hard cap on tool depth=2; if unmet acceptance after two rounds, escalate to human.
* Parallelize S2 (metrics+logs) with S3 (known issues) to hide latency.
* Latency SLOs propagate: if a tool is slow, scheduler downgrades scope (sample fewer logs, narrower time window).

**C) FP&A pack with hard cost ceiling**

* Use cached embeddings and artifact‑level summaries; disallow re‑embedding for the run.
* The Writer agent runs Medium for sections with low variance; for the CEO letter, enable Large via VOI check (high impact, moderate extra cost).
* Chart rendering is non‑LLM; keep LLM out of the compute‑heavy loop.

### 11.6 Telemetry & control loops

* **Live accounting:** record prompt/response tokens, wall time, per‑tool latency; expose “cost burn‑down” and “time remaining”.
* **Adaptive VOI threshold:** if over budget burn, raise `VOI_threshold` to curb upgrades; if under, allow selective upgrades on critical steps.
* **Caching:** prompt template + memory pack hashing to reuse previous completions; store *spec → artifact* maps for deterministic transforms.
* **Post‑run tuning:** regress quality vs. spend to calibrate `expected_quality_gain(step, m)`.

### 11.7 Guardrails for cost blowups

* **Circuit breaker:** if any single step consumes > X% of plan budget, halt and replan.
* **Quota enforcement:** per‑role daily token/cost caps with graceful degradation (“produce outline only”).
* **Fail‑closed:** on estimator failure or anomalous spend, fall back to Small + human gate.

---

### Checklists (quick apply)

**Memory routing**

* [ ] Lane taxonomy + schemas + TTLs
* [ ] Deterministic query builder (goal + acceptance + deps)
* [ ] Lexical→dense→re‑rank→budgeted pack pipeline
* [ ] Provenance required; “cite or drop” policy
* [ ] Utility logging (cited, validated) and lane weight learning

**Inter‑agent negotiation**

* [ ] Proposer/Critic/Judge roles with schemas
* [ ] Rubric + disqualifiers; bounded rounds
* [ ] Convergence tests (no‑new‑arguments, monotone scores)
* [ ] Weighted voting by reliability; dissent ledger
* [ ] Two‑phase commit for high‑risk changes

**Cost/latency governance**

* [ ] Cost/latency models per resource; SLOs per plan/step
* [ ] Coarse‑to‑fine gating with VOI escalation
* [ ] Context tightening before model upgrade
* [ ] Depth/recursion caps; parallelism where safe
* [ ] Live burn‑down; circuit breakers; caching strategy

These sections should slot directly into the earlier architecture, strengthening long‑horizon reliability while keeping the system measurable, governable, and cost‑aware.
