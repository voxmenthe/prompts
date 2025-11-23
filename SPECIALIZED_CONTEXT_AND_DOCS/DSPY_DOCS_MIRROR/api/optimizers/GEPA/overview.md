# dspy.GEPA: Reflective Prompt Optimizer

**GEPA** (Genetic-Pareto) is a reflective optimizer proposed in "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457)), that adaptively evolves _textual components_ (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with a text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. This allows GEPA to propose high performing prompts in very few rollouts.

## dspy.GEPA

```python
class GEPA(metric, *, auto=None, max_full_evals=None, max_metric_calls=None, reflection_minibatch_size=3, candidate_selection_strategy='pareto', reflection_lm=None, skip_perfect_score=True, add_format_failure_as_feedback=False, instruction_proposer=None, component_selector='round_robin', use_merge=True, max_merge_invocations=5, num_threads=None, failure_score=0.0, perfect_score=1.0, log_dir=None, track_stats=False, use_wandb=False, wandb_api_key=None, wandb_init_kwargs=None, track_best_outputs=False, warn_on_score_mismatch=True, use_mlflow=False, seed=0, gepa_kwargs=None)
```

GEPA is an evolutionary optimizer, which uses reflection to evolve text components
of complex systems. GEPA is proposed in the paper [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457).
The GEPA optimization engine is provided by the `gepa` package, available from [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa).

GEPA captures full traces of the DSPy module's execution, identifies the parts of the trace
corresponding to a specific predictor, and reflects on the behaviour of the predictor to
propose a new instruction for the predictor. GEPA allows users to provide textual feedback
to the optimizer, which is used to guide the evolution of the predictor. The textual feedback
can be provided at the granularity of individual predictors, or at the level of the entire system's
execution.

To provide feedback to the GEPA optimizer, implement a metric as follows:
```
def metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return {'score': float, 'feedback': str} corresponding 
    to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
    f"This trajectory got a score of {score}."
    """
    ...
```

GEPA can also be used as a batch inference-time search strategy, by passing `valset=trainset, track_stats=True, track_best_outputs=True`, and using the
`detailed_results` attribute of the optimized program (returned by `compile`) to get the Pareto frontier of the batch. `optimized_program.detailed_results.best_outputs_valset` will contain the best outputs for each task in the batch.

Example:
```
gepa = GEPA(metric=metric, track_stats=True)
batch_of_tasks = [dspy.Example(...) for task in tasks]
new_prog = gepa.compile(student, trainset=trainset, valset=batch_of_tasks)
pareto_frontier = new_prog.detailed_results.val_aggregate_scores
# pareto_frontier is a list of scores, one for each task in the batch.
```

Args:
    metric: The metric function to use for feedback and evaluation.
    auto: The auto budget to use for the run. Options: "light", "medium", "heavy".
    max_full_evals: The maximum number of full evaluations to perform.
    max_metric_calls: The maximum number of metric calls to perform.
    reflection_minibatch_size: The number of examples to use for reflection in a single GEPA step. Default is 3.
    candidate_selection_strategy: The strategy to use for candidate selection. Default is "pareto", 
        which stochastically selects candidates from the Pareto frontier of all validation scores. 
        Options: "pareto", "current_best".
    reflection_lm: The language model to use for reflection. Required parameter. GEPA benefits from 
        a strong reflection model. Consider using `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` 
        for optimal performance.
    skip_perfect_score: Whether to skip examples with perfect scores during reflection. Default is True.
    instruction_proposer: Optional custom instruction proposer implementing GEPA's ProposalFn protocol.
        **Default: None (recommended for most users)** - Uses GEPA's proven instruction proposer from 
        the [GEPA library](https://github.com/gepa-ai/gepa), which implements the 
        [`ProposalFn`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py). This default 
        proposer is highly capable and was validated across diverse experiments reported in the GEPA 
        paper and tutorials.

        See documentation on custom instruction proposers 
        [here](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/#custom-instruction-proposers).
        
        **Advanced Feature**: Only needed for specialized scenarios:
        - **Multi-modal handling**: Processing dspy.Image inputs alongside textual information
        - **Nuanced control over constraints**: Fine-grained control over instruction length, format, 
          and structural requirements beyond standard feedback mechanisms
        - **Domain-specific knowledge injection**: Specialized terminology or context that cannot be 
          provided through feedback_func alone
        - **Provider-specific prompting**: Optimizations for specific LLM providers (OpenAI, Anthropic) 
          with unique formatting preferences
        - **Coupled component updates**: Coordinated updates of multiple components together rather 
          than independent optimization
        - **External knowledge integration**: Runtime access to databases, APIs, or knowledge bases
        
        The default proposer handles the vast majority of use cases effectively. Use 
        MultiModalInstructionProposer() from dspy.teleprompt.gepa.instruction_proposal for visual 
        content or implement custom ProposalFn for highly specialized requirements.
        
        Note: When both instruction_proposer and reflection_lm are set, the instruction_proposer is called 
        in the reflection_lm context. However, reflection_lm is optional when using a custom instruction_proposer. 
        Custom instruction proposers can invoke their own LLMs if needed.
    component_selector: Custom component selector implementing the [ReflectionComponentSelector](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/base.py) protocol,
        or a string specifying a built-in selector strategy. Controls which components (predictors) are selected 
        for optimization at each iteration. Defaults to 'round_robin' strategy which cycles through components 
        one at a time. Available string options: 'round_robin' (cycles through components sequentially), 
        'all' (selects all components for simultaneous optimization). Custom selectors can implement strategies 
        using LLM-driven selection logic based on optimization state and trajectories. 
        See [gepa component selectors](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/component_selector.py) 
        for available built-in selectors and the ReflectionComponentSelector protocol for implementing custom selectors.
    add_format_failure_as_feedback: Whether to add format failures as feedback. Default is False.
    use_merge: Whether to use merge-based optimization. Default is True.
    max_merge_invocations: The maximum number of merge invocations to perform. Default is 5.
    num_threads: The number of threads to use for evaluation with `Evaluate`. Optional.
    failure_score: The score to assign to failed examples. Default is 0.0.
    perfect_score: The maximum score achievable by the metric. Default is 1.0. Used by GEPA 
        to determine if all examples in a minibatch are perfect.
    log_dir: The directory to save the logs. GEPA saves elaborate logs, along with all candidate 
        programs, in this directory. Running GEPA with the same `log_dir` will resume the run 
        from the last checkpoint.
    track_stats: Whether to return detailed results and all proposed programs in the `detailed_results` 
        attribute of the optimized program. Default is False.
    use_wandb: Whether to use wandb for logging. Default is False.
    wandb_api_key: The API key to use for wandb. If not provided, wandb will use the API key 
        from the environment variable `WANDB_API_KEY`.
    wandb_init_kwargs: Additional keyword arguments to pass to `wandb.init`.
    track_best_outputs: Whether to track the best outputs on the validation set. track_stats must 
        be True if track_best_outputs is True. The optimized program's `detailed_results.best_outputs_valset` 
        will contain the best outputs for each task in the validation set.
    warn_on_score_mismatch: GEPA (currently) expects the metric to return the same module-level score when 
        called with and without the pred_name. This flag (defaults to True) determines whether a warning is 
        raised if a mismatch in module-level and predictor-level score is detected.
    seed: The random seed to use for reproducibility. Default is 0.
    gepa_kwargs: (Optional) Additional keyword arguments to pass directly to [gepa.optimize](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py).
        Useful for accessing advanced GEPA features not directly exposed through DSPy's GEPA interface.
        
        Available parameters:
        - batch_sampler: Strategy for selecting training examples. Can be a [BatchSampler](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/batch_sampler.py) instance or a string 
          ('epoch_shuffled'). Defaults to 'epoch_shuffled'. Only valid when reflection_minibatch_size is None.
        - merge_val_overlap_floor: Minimum number of shared validation ids required between parents before 
          attempting a merge subsample. Only relevant when using `val_evaluation_policy` other than 'full_eval'. 
          Default is 5.
        - stop_callbacks: Optional stopper(s) that return True when optimization should stop. Can be a single 
          [StopperProtocol](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py) or a list of StopperProtocol instances. 
          Examples: [FileStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), 
          [TimeoutStopCondition](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), 
          [SignalStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), 
          [NoImprovementStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), 
          or custom stopping logic. Note: This overrides the default 
          max_metric_calls stopping condition.
        - use_cloudpickle: Use cloudpickle instead of pickle for serialization. Can be helpful when the 
          serialized state contains dynamically generated DSPy signatures. Default is False.
        - val_evaluation_policy: Strategy controlling which validation ids to score each iteration. Can be 
          'full_eval' (evaluate every id each time) or an [EvaluationPolicy](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/eval_policy.py) instance. Default is 'full_eval'.
        - use_mlflow: If True, enables MLflow integration to log optimization progress. 
          MLflow can be used alongside Weights & Biases (WandB).
        - mlflow_tracking_uri: The tracking URI to use for MLflow (when use_mlflow=True).
        - mlflow_experiment_name: The experiment name to use for MLflow (when use_mlflow=True).
        
        Note: Parameters already handled by DSPy's GEPA class will be overridden by the direct parameters 
        and should not be passed through gepa_kwargs.
    
Note:
    Budget Configuration: Exactly one of `auto`, `max_full_evals`, or `max_metric_calls` must be provided.
    The `auto` parameter provides preset configurations: "light" for quick experimentation, "medium" for
    balanced optimization, and "heavy" for thorough optimization.
    
    Reflection Configuration: The `reflection_lm` parameter is required and should be a strong language model.
    GEPA performs best with models like `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)`.
    The reflection process analyzes failed examples to generate feedback for program improvement.
    
    Merge Configuration: GEPA can merge successful program variants using `use_merge=True`.
    The `max_merge_invocations` parameter controls how many merge attempts are made during optimization.
    
    Evaluation Configuration: Use `num_threads` to parallelize evaluation. The `failure_score` and 
    `perfect_score` parameters help GEPA understand your metric's range and optimize accordingly.
    
    Logging Configuration: Set `log_dir` to save detailed logs and enable checkpoint resuming.
    Use `track_stats=True` to access detailed optimization results via the `detailed_results` attribute.
    Enable `use_wandb=True` for experiment tracking and visualization.
    
    Reproducibility: Set `seed` to ensure consistent results across runs with the same configuration.


### auto_budget

```python
def auto_budget(self, num_preds, num_candidates, valset_size, minibatch_size=35, full_eval_steps=5)
```

### compile

```python
def compile(self, student, *, trainset, teacher=None, valset=None)
```

GEPA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.
If no valset is provided, GEPA will use the trainset for both.

Parameters:
- student: The student module to optimize.
- trainset: The training set to use for reflective updates.
- valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/gepa/gepa.py` (lines 148–598)


One of the key insights behind GEPA is its ability to leverage domain-specific textual feedback. Users should provide a feedback function as the GEPA metric, which has the following call signature:
## dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric

```python
class GEPAFeedbackMetric
```

### __call__

```python
def __call__(gold, pred, trace, pred_name, pred_trace)
```

This function is called with the following arguments:
- gold: The gold example.
- pred: The predicted output.
- trace: Optional. The trace of the program's execution.
- pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
    the feedback is being requested.
- pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding 
to the predictor.
If not available at the predictor level, the metric can also return a text feedback at the program level
(using just the gold, pred and trace).
If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
f"This trajectory got a score of {score}."

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/gepa/gepa.py` (lines 26–53)


When `track_stats=True`, GEPA returns detailed results about all of the proposed candidates, and metadata about the optimization run. The results are available in the `detailed_results` attribute of the optimized program returned by GEPA, and has the following type:
## dspy.teleprompt.gepa.gepa.DspyGEPAResult

```python
class DspyGEPAResult
```

Additional data related to the GEPA run.

Fields:
- candidates: list of proposed candidates (component_name -> component_text)
- parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
- val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
- val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
- per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the best score on t
- discovery_eval_counts: Budget (number of metric calls / rollouts) consumed up to the discovery of each candidate

- total_metric_calls: total number of metric calls made across the run
- num_full_val_evals: number of full validation evaluations performed
- log_dir: where artifacts were written (if any)
- seed: RNG seed for reproducibility (if known)

- best_idx: candidate index with the highest val_aggregate_scores
- best_candidate: the program text mapping for best_idx

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/gepa/gepa.py` (lines 57–145)


## Usage Examples

See GEPA usage tutorials in [GEPA Tutorials](../../../tutorials/gepa_ai_program/index.md).

### Inference-Time Search

GEPA can act as a test-time/inference search mechanism. By setting your `valset` to your _evaluation batch_ and using `track_best_outputs=True`, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search.

```python
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
highest_score_achieved_per_task = new_prog.detailed_results.highest_score_achieved_per_val_task
best_outputs = new_prog.detailed_results.best_outputs_valset
```

## How Does GEPA Work?

### 1. **Reflective Prompt Mutation**

GEPA uses LLMs to _reflect_ on structured execution traces (inputs, outputs, failures, feedback), targeting a chosen module and proposing a new instruction/program text tailored to real observed failures and rich textual/environmental feedback.

### 2. **Rich Textual Feedback as Optimization Signal**

GEPA can leverage _any_ textual feedback available—not just scalar rewards. This includes evaluation logs, code traces, failed parses, constraint violations, error message strings, or even isolated submodule-specific feedback. This allows actionable, domain-aware optimization. 

### 3. **Pareto-based Candidate Selection**

Rather than evolving just the _best_ global candidate (which leads to local optima or stagnation), GEPA maintains a Pareto frontier: the set of candidates which achieve the highest score on at least one evaluation instance. In each iteration, the next candidate to mutate is sampled (with probability proportional to coverage) from this frontier, guaranteeing both exploration and robust retention of complementary strategies.

### Algorithm Summary

1. **Initialize** the candidate pool with the the unoptimized program.
2. **Iterate**:
   - **Sample a candidate** (from Pareto frontier).
   - **Sample a minibatch** from the train set.
   - **Collect execution traces + feedbacks** for module rollout on minibatch.
   - **Select a module** of the candidate for targeted improvement.
   - **LLM Reflection:** Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback.
   - **Roll out the new candidate** on the minibatch; **if improved, evaluate on Pareto validation set**.
   - **Update the candidate pool/Pareto frontier.**
   - **[Optionally] System-aware merge/crossover**: Combine best-performing modules from distinct lineages.
3. **Continue** until rollout or metric budget is exhausted. 
4. **Return** candidate with best aggregate performance on validation.

## Implementing Feedback Metrics

A well-designed metric is central to GEPA's sample efficiency and learning signal richness. GEPA expects the metric to returns a `dspy.Prediction(score=..., feedback=...)`. GEPA leverages natural language traces from LLM-based workflows for optimization, preserving intermediate trajectories and errors in plain text rather than reducing them to numerical rewards. This mirrors human diagnostic processes, enabling clearer identification of system behaviors and bottlenecks.

Practical Recipe for GEPA-Friendly Feedback:

- **Leverage Existing Artifacts**: Use logs, unit tests, evaluation scripts, and profiler outputs; surfacing these often suffices.
- **Decompose Outcomes**: Break scores into per-objective components (e.g., correctness, latency, cost, safety) and attribute errors to steps.
- **Expose Trajectories**: Label pipeline stages, reporting pass/fail with salient errors (e.g., in code generation pipelines).
- **Ground in Checks**: Employ automatic validators (unit tests, schemas, simulators) or LLM-as-a-judge for non-verifiable tasks (as in PUPA).
- **Prioritize Clarity**: Focus on error coverage and decision points over technical complexity.

### Examples

- **Document Retrieval** (e.g., HotpotQA): List correctly retrieved, incorrect, or missed documents, beyond mere Recall/F1 scores.
- **Multi-Objective Tasks** (e.g., PUPA): Decompose aggregate scores to reveal contributions from each objective, highlighting tradeoffs (e.g., quality vs. privacy).
- **Stacked Pipelines** (e.g., code generation: parse → compile → run → profile → evaluate): Expose stage-specific failures; natural-language traces often suffice for LLM self-correction.

## Custom Instruction Proposal

For advanced customization of GEPA's instruction proposal mechanism, including custom instruction proposers and component selectors, see [Advanced Features](GEPA_Advanced.md).

## Further Reading

- [GEPA Paper: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [GEPA Github](https://github.com/gepa-ai/gepa) - This repository provides the core GEPA evolution pipeline used by `dspy.GEPA` optimizer.
- [DSPy Tutorials](../../../tutorials/gepa_ai_program/index.md)
