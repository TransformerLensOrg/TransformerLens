# Plan & Code Review Specification

## Execution Model

### Findings Accumulator

Maintain a running findings list across all phases. Each phase appends to it
using `[P0]`, `[P1]`, etc. tags. The final output is produced FROM this list,
not from memory. This prevents forgetting earlier findings and eliminates
redundant re-verification.

### Phase 0: HF Source Reference (do this once)

Before any evaluation, read the HuggingFace Transformers source and produce a
structured reference block. All subsequent phases reference this block instead
of re-reading the source independently.

Record: module paths, config fields, forward pass flow, bias presence,
normalization details, attention type, MLP type.

**Exit criteria:** Every architectural property has a recorded module path,
config field, or source line number.

---

## Phase 1: Factual Verification

Every concrete claim in the plan or code must be verified against the Phase 0
reference and source code. Never trust that a description of existing code is
accurate.

### For architecture/model claims:
- Cross-reference every module path against the P0 reference
- Verify every config attribute name matches what the code actually uses
- Verify parameter shapes, bias presence, and dimensional relationships
- Verify the forward pass flow matches what the plan/code describes

### For claims about existing codebase infrastructure:
- Read the referenced files at the cited line numbers
- Verify function signatures, class hierarchies, and calling conventions
- Verify that claimed capabilities actually exist (e.g., "JointQKVAttentionBridge accepts split_qkv_matrix" — read the __init__ to confirm)
- Verify inheritance chains — does the class actually inherit the mixin/method the plan assumes?

### For test claims:
- Verify test files exist, check skip/xfail markers, count actual test methods
- Verify claimed tolerances match actual code
- Verify claimed CI behavior by reading workflow files

**Exit criteria:** Every concrete claim has a verified or refuted annotation.

---

## Phase 2: Numerical/Semantic Correctness

When a plan reimplements a computation or defines weight conversions, diff
against the Phase 0 reference line by line:

- **Operation order**: Does split happen before or after layernorm? Before or after reshape?
- **Dimension handling**: Are transpose/reshape dimensions correct? Does `view(-1, ...)` infer the right dimension?
- **Dtype behavior**: Does HF upcast softmax to fp32? Does the bridge match?
- **Conditional logic**: Does HF only pad under certain conditions (flash attention)? Does the bridge match those conditions?
- **Config attribute access**: Is the bridge reading from the right object (HF config vs HF module instance vs bridge config)?
- **Return values**: Does the bridge return the same tuple structure as HF?

**Exit criteria:** Every weight conversion key and rearrange pattern has been
checked against the P0 forward pass and dimension records.

---

## Phase 3: Design Evaluation

### Over-engineering detection:
- Count concrete implementations vs abstractions. If an abstraction has only 1-2 implementations today, it's probably premature. Prefer the simplest approach that works now.
- Watch for protocol/interface designs justified by "future architectures" that don't exist yet.
- Watch for configuration flexibility that no current use case exercises.

### Under-specification detection:
- Any fix described as "ensure X" or "handle Y correctly" without specifying *how* is underspecified. There should be a concrete approach or at minimum enumerated options with tradeoffs.
- Any claim about behavior ("hooks fire correctly") without specifying the mechanism that makes it happen.

### Dependency and ordering analysis:
- Can the proposed order actually work? Does step N depend on outputs of step N+2?
- Are investigative tasks (unknown scope) mixed with mechanical tasks (known scope) in the same work unit? If so, the mechanical tasks may be blocked unnecessarily.
- Do multiple steps touch the same files? If so, should they be merged or sequenced explicitly?

**Exit criteria:** Every design decision that deviates from existing patterns
has been evaluated and annotated.

---

## Phase 4: Completeness & Edge Cases

### What happens when it fails?
- If a user accesses an unsupported property, do they get a clear error or a confusing crash?
- If a config variant isn't supported, does the adapter raise NotImplementedError or silently produce wrong results?
- If a submodule doesn't exist on a particular layer, does setup crash or skip gracefully?

### What's missing?
- Are all files that need modification listed?
- Are all new exports/registrations covered?
- Is there a verification strategy that tests what the plan claims to fix?
- Does the test strategy use models that are actually available (CI-cached, programmatic, or small enough to download)?

### Inventory accuracy:
- When the plan says "N files" or "N tests", count the actual files. Plans frequently undercount.

**Exit criteria:** Every required file, registration, and edge case has been
checked and annotated.

---

## Phase 5: Differential Review (for completed code)

Phase 5 is NOT a re-run of Phase 1. Its scope is strictly:

### Findings resolution:
- Walk the entire findings accumulator from P0-P4. Confirm each finding is
  either resolved in the code or documented as accepted.
- Flag any finding that was neither fixed nor acknowledged.

### Plan-to-code match:
- Every task in the plan should have corresponding code changes. Flag anything
  planned but not implemented, or implemented but not planned.

### Cross-phase issues:
- Flag any new discrepancies that only emerge when seeing plan + code + HF
  source together — things no single phase would catch in isolation.

### Test quality:
- Are assertions meaningful (testing actual behavior) or tautological (testing mock setup)?
- Are tolerances justified by observed values, not picked arbitrarily?
- Do tests exercise both the happy path and the edge cases the plan identified?
- Are test fixtures appropriately scoped (session for model loading, function for mutations)?

**Exit criteria:** Every P0-P4 finding has a resolution status, every plan
item has a corresponding code change, and test quality has been assessed.

---

## Meta-principles

1. **Verify, don't trust.** Read the actual code for every claim. Plans written from memory are frequently wrong about attribute names, parameter counts, and conditional logic.

2. **Severity triage.** Not every issue is equal. A wrong attribute name is CRITICAL (crashes at runtime). An over-designed abstraction is MEDIUM (works but adds maintenance cost). A documentation gap is LOW. Label accordingly.

3. **One issue per finding.** Don't bundle "the LN name is wrong and also the tolerance is loose" — these are separate findings with separate severities.

4. **Distinguish "wrong" from "could be better."** A plan that works but isn't optimal is fine. A plan that will crash or produce silently wrong results is not. Focus review energy on correctness first, then design.

5. **Track across iterations.** On re-reviews, explicitly confirm each previous finding was addressed before looking for new issues. Don't re-raise resolved items.

6. **Don't invent requirements.** Review against what the plan says it will do, not what you think it should do. If the plan explicitly defers something, that's a valid choice — flag only if the deferral creates a silent failure path.
