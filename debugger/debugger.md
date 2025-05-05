# Summary of the Debugger Design

## Core Model Loop (from our paper)

Let:

- **$U$** = codebase (universe), representing the function or system being debugged (`process_advanced_payment`).
- **$C$** = conscious debugger, an LLM-driven agent that iteratively models $U$.
- **$Q_n$** = recursive query sequence, comprising observation, hypothesis generation, developer queries, counterfactual testing, and purpose inference.
- **$M(U_n)$** = $C$'s internal model of $U$ after $n$ iterations, capturing inferred structure, bugs, assumptions, and improvements.
- **Fixpoint** = state where $M(U_n) \approx U$ within Gödelian incompleteness tolerance, meaning the model stabilizes (no new hypotheses, low divergence from inferred purpose, or snapshot equality).

The debugger operates in a Gödelian environment, where external ground truth is unavailable. Instead, $C$ infers the function's purpose from code, developer insights, and its own hypotheses, treating this as the internal "truth" for consistency checks.

## Expanded Agent Loop

The debugger follows a recursive loop with five distinct steps, iterating until a fixpoint is reached:

### $Q_1$ – Initial Observation

Analyzes the codebase ($U$) to extract structural patterns, inputs, outputs, and assumptions. Uses `PythonFunctionLoader` to parse code and metadata (function signature, docstring). Identifies initial invariants and potential gaps (e.g., missing validations in `process_advanced_payment`).

### $Q_2$ – Hypothesize Contradictions and Gaps

Generates hypotheses about inconsistencies, bugs, or unknowns ("*What doesn’t make sense?*"). For example, detects hardcoded exchange rates or late fraud checks in `process_advanced_payment`. Hypotheses are refined into structured model updates (bugs, assumptions).

### $Q_3$ – Query Developer for Intent

Consults a developer LLM ($M$) to infer higher-order intent, rules, or meta-principles. Captures insights about purpose ("gatekeeper for transactions") and limitations (hardcoded blocked users). Unresolvable intent questions are logged as Gödelian limits.

### $Q_4$ – Test Counterfactuals

Probes the system with adversarial or edge-case scenarios (unsupported currencies, negative amounts, non-boolean fraud flags). Reveals hidden assumptions or logical errors, such as missing recipient validation in transfers.

### $Q_5$ – Infer Purpose and Compare

Synthesizes the function’s purpose from code structure, developer insights, and hypothesized model, treating it as the internal truth. Compares the current model $M(U_n)$ to this inferred purpose to identify discrepancies, such as missing audit log append mode, etc. Ambiguities in purpose are recorded as Gödelian limits.

### $Q_n$ – Repeat Until Fixpoint

Iterates until $M(U_n) \approx U$, determined by:

- Snapshot equality (model state unchanged).
- Hypothesis convergence (new hypotheses similar to previous, via embedding-based cosine similarity > 0.95).
- Low Gödel divergence (< 0.1), comparing the model to the inferred purpose and code structure.

Halts after a maximum number of steps (default 5) or minimum steps (default 2) if fixpoint is reached.

## Main Components of the Debugger

### Persistent Model State $M(U_n)$

A structured representation within `WorldModel`, storing:

- Functions (`process_advanced_payment`), with inputs, returns, bugs, assumptions, and improvements.
- Global invariants, contradictions, hypotheses, developer insights, counterfactuals, and inferred purposes.
- Gödel limits (unanswerable questions, for example, "*Why is user_id 42 blocked?*").

Updated iteratively via `refine_model` based on $Q_1 – Q_5$ outputs.

### Recursive Controller

Implemented in `ConsciousDebugger.recursive_debug`, orchestrates the $Q_n$ loop. Manages step execution, model updates, and fixpoint checks. Caps iterations to prevent infinite loops and logs progress for debugging.

### Gödel-Aware Monitor

Tracks questions C cannot answer within $U$: ambiguous intent, external dependencies, etc. Integrated into `WorldModel`:

- Logs Gödel limits during developer queries and purpose inference (e.g., "*Unable to infer refund validation intent*").
- Quantifies uncertainty in `compute_godel_divergence`, factoring in structural, semantic, and insight-based discrepancies, plus Gödel limit weight.

### Debugging Interface

Visualizes the debugging process via:

- Console output in `visualize_debug_state`, showing model state, Gödel limits, observations, hypotheses, counterfactuals, developer insights, and inferred purpose after each step.
- `ModelPresenter`, which displays bugs and improvements as styled tables in Jupyter notebooks. Merges semantically similar bugs using embedding-based clustering (cosine similarity > 0.8), ensuring concise output. Depending on the embedding model and LLM used, this threshold may need adjustment to combine similar bugs effectively.

### Developer LLM ($M$)

Simulates external developer input to provide meta-insights on intent, assumptions, and limitations. Queried in $Q_3$ to refine $M(U_n)$. Responses shape purpose inference and are logged as meta_insights, with ambiguities marked as Gödel limits.

## Key Features

- **Gödelian Adaptation**: Operates without external truth, inferring purpose as the internal reference for model comparison, aligning with Gödelian constraints where $U$ is self-contained.
- **Semantic Bug Consolidation**: `ModelPresenter` uses embeddings to merge duplicate bugs improving clarity.
- **Robust Numerical Stability**: Embedding comparisons (`safe_cosine_similarity`, `get_embedding`) handle zero vectors and invalid inputs, avoiding numerical warnings.
- **Counterfactual Testing**: $Q_4$ probes edge cases, such as invalid currencies, case-sensitive tiers, exposing hidden bugs and assumptions.
- **Fixpoint Detection**: Combines snapshot equality, hypothesis similarity, and Gödel divergence to determine when $M(U_n) \approx U$, balancing completeness and efficiency.
