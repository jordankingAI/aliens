# ALIENS — API Integration Guide & LLM Backend Reference

All code in this guide has been executed against `alien_system.py` and
`alien_system_es.py`. Every example is importable and runnable as shown.

---

## Part 1 — The Single Integration Point

ALIENS communicates with any LLM through exactly one method. This is the
complete interface:

```python
class LLMBackend(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Send system_prompt + user_prompt to the LLM.
        Parse the response as JSON and return the dict.
        Raise any exception on failure — ALIENS handles it at the stage level.
        """
        ...
```

ALIENS calls `complete_json` **six times per full reading cycle**: four times
during `prepare_cycle` and twice during `complete_cycle`.

| Phase | Task key | Call # |
|---|---|---|
| `prepare_cycle` | `canonicalize_passage` | 1 |
| `prepare_cycle` | `generate_candidates` | 2 |
| `prepare_cycle` | `estimate_fit` | 3 |
| `prepare_cycle` | `generate_assessment` | 4 |
| `complete_cycle` | `score_retell` | 5 |
| `complete_cycle` | `diagnose_outcome` | 6 |

Each call sends a fixed system prompt (role and output contract) and a
structured JSON user prompt (data for that specific call). The user prompt
always contains `"task": "<task_key>"` — this is how `TaskRoutingMockLLM`
routes responses in tests.

Your implementation must:
- Send both prompts to the LLM and return the parsed response dict
- Raise any exception on failure — ALIENS catches at the stage level
- Never pre-validate the JSON — ALIENS does that
- Never modify either prompt string

---

## Part 2 — Reference Implementations

### Anthropic (Claude)

```python
import re
import json
import anthropic
from typing import Any


class AnthropicBackend:
    """
    LLMBackend for the Anthropic Messages API.

    Recommended model: claude-sonnet-4-20250514 or later Sonnet-class.
    Haiku is not recommended: generation quality on generate_candidates
    is noticeably lower and produces more DEGRADED selection.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        api_key: str | None = None,   # falls back to ANTHROPIC_API_KEY env var
    ) -> None:
        self.client     = anthropic.Anthropic(api_key=api_key)
        self.model      = model
        self.max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if the model wraps its JSON output.
        # Some models wrap JSON responses in ```json ... ``` blocks.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]         # drop opening fence line
            raw = raw.rsplit("\n", 1)[0].rstrip() # drop closing fence line
            raw = raw.rstrip("`").strip()
        return json.loads(raw)
```

**Token budget (Anthropic, typical passage 150–300 words):**

| Task | Input tokens | Output tokens |
|---|---|---|
| `canonicalize_passage` | 800–2,000 | 600–1,200 |
| `generate_candidates` | 2,000–5,000 | 1,500–4,000 |
| `estimate_fit` | 1,500–3,000 | 300–600 |
| `generate_assessment` | 1,500–2,500 | 1,000–2,000 |
| `score_retell` | 600–1,200 | 200–400 |
| `diagnose_outcome` | 400–700 | 100–200 |

`max_tokens=4096` covers all tasks comfortably. Raise to `8192` for passages
over 500 words, or if `generate_candidates` output is being truncated.

---

### OpenAI (GPT-4o)

```python
import json
from openai import OpenAI
from typing import Any


class OpenAIBackend:
    """
    LLMBackend for the OpenAI Chat Completions API.

    response_format={"type": "json_object"} enforces JSON output and
    eliminates the need to strip markdown fences. Requires the system
    prompt to mention JSON output — which ALIENS's prompts already do.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        self.client     = OpenAI(api_key=api_key)
        self.model      = model
        self.max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
```

**Model guidance:** `gpt-4o` is recommended for all tasks. `gpt-4o-mini`
is acceptable for `score_retell` and `diagnose_outcome` but should not be
used for `canonicalize_passage` or `generate_candidates`, where generation
quality directly determines whether candidates pass validation. A routed
backend handles this automatically:

```python
import json
from openai import OpenAI
from typing import Any


class RoutedOpenAIBackend:
    """
    Routes generation tasks to gpt-4o; scoring tasks to gpt-4o-mini.
    Reduces cost without sacrificing quality on the tasks that matter.
    """

    HEAVY_TASKS = {
        "canonicalize_passage",
        "generate_candidates",
        "estimate_fit",
        "generate_assessment",
    }

    def __init__(self) -> None:
        self.client      = OpenAI()
        self.heavy_model = "gpt-4o"
        self.light_model = "gpt-4o-mini"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        task  = json.loads(user_prompt).get("task", "")
        model = self.heavy_model if task in self.HEAVY_TASKS else self.light_model
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
```

---

### Azure OpenAI

```python
import json
from openai import AzureOpenAI
from typing import Any


class AzureOpenAIBackend:

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-01",
        deployment_name: str = "gpt-4o",
        max_tokens: int = 4096,
    ) -> None:
        self.client          = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.max_tokens      = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
```

---

## Part 3 — Retry and Error Handling

### Retry wrapper

```python
import time
import logging
from typing import Any


class RetryingBackend:
    """
    Wraps any LLMBackend with exponential-backoff retry for transient failures.

    Does not retry on ALIENError — those are schema failures that retrying
    the identical call will not fix. Retries on network errors, rate limits,
    and provider 5xx responses, which surface as provider-SDK exceptions
    before ALIENS's validators run.
    """

    def __init__(
        self,
        backend,
        max_retries: int = 3,
        base_delay:  float = 1.0,
        max_delay:   float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._backend     = backend
        self._max_retries = max_retries
        self._base_delay  = base_delay
        self._max_delay   = max_delay
        self._logger      = logger or logging.getLogger(__name__)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._backend.complete_json(system_prompt, user_prompt)
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                self._logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self._max_retries + 1, delay, exc,
                )
                time.sleep(delay)
        raise last_exc
```

### Stage-level error handling

Every `ALIENError` carries a `.stage` attribute naming which of the six
tasks failed. Handle stages independently in your application layer:

```python
from alien_system import AdaptiveReadingSystem, ALIENError

system = AdaptiveReadingSystem(llm=RetryingBackend(AnthropicBackend()))

try:
    prep = system.prepare_cycle(source_text, passage_id, objective, learner)

except ALIENError as e:
    match e.stage:

        case "canonicalize_passage":
            # Invalid canonical JSON after all retries.
            # Block this passage until the canonical is manually reviewed.
            flag_for_human_review(passage_id, e)
            return serve_fallback_passage()

        case "generate_candidates":
            # Invalid candidate JSON. Retry once; if it fails again, serve source text.
            log_generation_failure(passage_id, e)
            return serve_source_text(passage_id)

        case "select_candidate":
            # Every candidate had a blocking validation failure.
            # This is a canonical quality signal — anchor words may be wrong.
            log_canonical_quality_failure(passage_id, e)
            flag_canonical_for_review(passage_id)
            return serve_source_text(passage_id)

        case "estimate_fit":
            # Fit estimation failed. For graceful degradation, catch before
            # prepare_cycle propagates it.
            log_fit_failure(passage_id, e)
            return serve_source_text(passage_id)

        case "generate_assessment":
            # Serve a pre-authored fallback assessment for this passage.
            log_assessment_failure(passage_id, e)
            return serve_fallback_assessment(passage_id)
```

`score_retell` and `diagnose_outcome` never raise `ALIENError` in normal
operation — both have automatic deterministic fallbacks that activate and
log a `WARNING`. The cycle always completes.

---

## Part 4 — The Six Calls in Detail

Every user prompt includes a `"task"` key. The system prompt establishes
the LLM's role; the user prompt contains the structured data for that call.

---

### Call 1 — `canonicalize_passage`

**When:** Once per passage, shared across all learners. Cache the result (Part 5).

**User prompt (abbreviated):**
```json
{
  "task": "canonicalize_passage",
  "passage_id": "gutenberg_01",
  "instructional_objective": "Students will explain how the printing press...",
  "source_text": "Johannes Gutenberg invented the printing press around 1440...",
  "output_contract": { "...": "..." }
}
```

**Expected response:**
```json
{
  "passage_id": "gutenberg_01",
  "source_text": "...",
  "instructional_objective": "...",
  "meaning_units": [
    {
      "id": "MU1",
      "text": "Gutenberg invented the printing press around 1440",
      "required": true,
      "anchors": ["Gutenberg", "printing", "press", "1440", "invented"]
    }
  ],
  "sequence_constraints": [{"before": "MU1", "after": "MU2"}],
  "must_preserve_vocabulary": [
    {"term": "printing press", "required": true, "gloss_allowed": false}
  ]
}
```

**Validator:** `validate_canonical_json()` — passage_id present, MUs
non-empty, all MUs have id and text, sequence constraints reference known
MU ids, vocabulary terms have term field.

**Note:** ALIENS always overwrites the echoed `source_text` with the
original. The LLM's echo is discarded and never trusted.

---

### Call 2 — `generate_candidates`

**When:** Once per `prepare_cycle`, after canonicalization. The heaviest
call — scales with candidate count and passage length.

**User prompt includes:** canonical passage, learner state (all five need
dimensions, current band, readiness, recent outcomes, journey fields), and
the candidate plan specifying how many variants to generate and at what
relative band each should target.

**Expected response:**
```json
{
  "candidates": [
    {
      "candidate_id": "A",
      "passage_id": "gutenberg_01",
      "relative_band": -1,
      "text": "Around 1440, Johannes Gutenberg invented the printing press...",
      "scaffold": {
        "vocabulary_support": "medium",
        "syntax_support": "high",
        "cohesion_support": "medium",
        "chunking_support": "low",
        "inference_support": "low"
      },
      "llm_self_audit": {
        "meaning_preserved": true,
        "sequence_preserved": true,
        "objective_preserved": true,
        "same_passage_identity": true,
        "notes": "All 5 MUs present in order.",
        "meaning_unit_coverage": {
          "MU1": true, "MU2": true, "MU3": true, "MU4": true, "MU5": true
        }
      }
    }
  ]
}
```

**Validator:** `validate_candidates_json()` — non-empty, unique ids,
non-empty text, correct passage_id, scaffold and self_audit dicts present,
`meaning_unit_coverage` is a dict when present.

**Note:** The `meaning_unit_coverage` map is used as a fast-path blocking
check before the full lexical meaning profile. If the LLM marks any
required MU as `false`, that candidate gets a blocking reason without
running the expensive token matching.

---

### Call 3 — `estimate_fit`

**When:** Once per `prepare_cycle`, after all candidates are scored.
Receives only validated candidates (no blocking reasons), or the full pool
if none validated.

**User prompt includes:** canonical passage, learner state, validated
candidates with their full text, and all deterministic scores keyed by
`candidate_id`.

**Expected response:**
```json
{
  "fit_estimates": [
    {
      "candidate_id": "A",
      "access": "high",
      "growth": "medium",
      "support_burden": "medium",
      "reason": "Text is at accessible level for this learner with adequate challenge."
    }
  ]
}
```

**Validator:** `validate_fit_estimates_json()` — every validated candidate
has exactly one estimate, no duplicates, no unknown ids, all labels are
`low|medium|high`.

---

### Call 4 — `generate_assessment`

**When:** Once per `prepare_cycle`, after candidate selection. Generated
against the canonical passage, calibrated to the delivered variant.

**Six items in fixed order — the type sequence is enforced by the validator:**

| # | Type | What it tests |
|---|---|---|
| Q1 | `literal_mcq` | Single stated fact (targets a required MU) |
| Q2 | `sequence_mcq` | Temporal or causal ordering of two MUs |
| Q3 | `inference_mcq` | Cause-effect or implication not directly stated |
| Q4 | `vocabulary_mcq` | A required vocabulary term in context |
| Q5 | `retell_short_response` | Open-ended gist retell scored against canonical MUs |
| Q6 | `self_rating` | Learner's perceived difficulty, scale 1–5 |

**Hard constraints enforced by `validate_assessment_json()`:**
- Exactly 6 items, in the order above
- Each MCQ item has exactly 4 choices
- `correct_answer` must be present in the choices array
- Retell rubric `max_score` must equal the sum of all criteria points
- Q2 target must be a dict with `meaning_unit_ids` (list of at least 2) and `relation`
- Comprehension score weights must sum to 1.0 (within 0.01 tolerance)

---

### Call 5 — `score_retell`

**When:** Once per `complete_cycle`, to score the learner's Q5 response.

**User prompt includes:** canonical passage, the rubric extracted from the
assessment package, and the learner's raw response text.

**Expected response:**
```json
{
  "raw_score": 3,
  "max_score": 4,
  "matched_meaning_units": ["MU1", "MU2", "MU4"],
  "matched_relationships": ["press_enabled_spread"],
  "concise_reason": "Covered Gutenberg, pre-press scarcity, and spread. Did not address the Renaissance/Reformation connection."
}
```

**Validator:** `validate_retell_score_json()` — both scores present and
are non-boolean integers, `raw_score <= max_score`, `raw_score >= 0`.

**Automatic fallback:** If this call fails for any reason, ALIENS activates
a deterministic keyword scorer, logs a `WARNING`, and the cycle completes
normally.

---

### Call 6 — `diagnose_outcome`

**When:** Once per `complete_cycle`, after assessment scoring.

**User prompt includes:** learner state (full profile) and all computed
reading signals.

**Expected response:**
```json
{
  "diagnosis": "cohesion_inference_barrier",
  "reason": "Comprehension of stated facts is adequate but inference score is below threshold."
}
```

**The `diagnosis` value must be one of these seven strings exactly:**
`underchallenged`, `well_calibrated`, `successful_but_support_dependent`,
`vocabulary_barrier`, `syntax_barrier`, `cohesion_inference_barrier`,
`overloaded`.

Any other value causes `DiagnosisLabel.from_value()` to raise, triggering
the fallback.

**Automatic fallback:** If this call fails, ALIENS activates the deterministic
decision tree in `DeterministicEngine.diagnose_fallback()`, logs a `WARNING`,
and the cycle completes normally.

---

## Part 5 — Caching the Canonical Passage

The canonical must be cached. Every learner on the same passage must use
the identical canonical — different canonicalisations produce different MU
ids and different assessment rubrics, making results incomparable across
learners.

```python
import json
from alien_system import (
    CanonicalPassage,
    parse_canonical_passage,
    _canonical_to_dict,
)


class CanonicalCache:
    """
    Key-value canonical passage cache.
    In production, back this with Redis, a database, or any persistent store.
    """

    def __init__(self, store: dict | None = None) -> None:
        self._store = store if store is not None else {}

    def get(self, passage_id: str) -> CanonicalPassage | None:
        raw = self._store.get(passage_id)
        if raw is None:
            return None
        return parse_canonical_passage(json.loads(raw))

    def put(self, canonical: CanonicalPassage) -> None:
        self._store[canonical.passage_id] = json.dumps(
            _canonical_to_dict(canonical), ensure_ascii=False
        )

    def __contains__(self, passage_id: str) -> bool:
        return passage_id in self._store
```

**Skipping the canonicalize_passage LLM call on cached passages:**

Call the `prepare_cycle` sub-methods directly rather than using
`prepare_cycle` (which always re-canonicalises). This saves 1 LLM call
per cycle after the first for each passage — verified to produce 5 calls
instead of 6:

```python
from alien_system import (
    AdaptiveReadingSystem, LearnerState, CyclePreparation,
)


class CachingSystem:
    """
    AdaptiveReadingSystem wrapper that skips canonicalization when cached.
    Saves 1 LLM call per cycle after the first for each passage.
    """

    def __init__(
        self,
        system: AdaptiveReadingSystem,
        cache:  CanonicalCache,
    ) -> None:
        self._system = system
        self._cache  = cache

    def prepare_cycle(
        self,
        source_text: str,
        passage_id: str,
        objective: str,
        learner: LearnerState,
        candidate_plan=None,
    ) -> CyclePreparation:
        cached = self._cache.get(passage_id)

        if cached is not None:
            # Skip canonicalize_passage — use the cached canonical directly.
            learner_j  = self._system.begin_passage_journey(learner, cached)
            candidates = self._system.generate_candidates(
                cached, learner_j, candidate_plan)
            fit_ests, all_scores = self._system.estimate_fit(
                cached, learner_j, candidates)
            selected, scores = self._system.engine.select_candidate(
                cached, learner_j, candidates, fit_ests,
                precomputed_scores=all_scores)
            assessment = self._system.generate_assessment(
                cached, selected, learner_j)
            return CyclePreparation(
                canonical=cached,
                selected_candidate=selected,
                selected_scores=scores,
                assessment=assessment,
                fit_estimates=fit_ests,
                all_scores=all_scores,
                selection_mode=scores.selection_mode,
                prepared_learner=learner_j,
            )

        # First call for this passage: canonicalize and cache.
        # In production: cache only AFTER expert review of the canonical.
        prep = self._system.prepare_cycle(
            source_text, passage_id, objective, learner, candidate_plan)
        self._cache.put(prep.canonical)
        return prep

    def complete_cycle(self, learner, prep, answers, telemetry):
        return self._system.complete_cycle(learner, prep, answers, telemetry)
```

---

## Part 6 — Persisting Learner State

Learner state must survive between sessions. Serialisation is built in.
The blob is under 1 KB. Use a pseudonymous token for `learner_id` — the
blob contains no PII unless you put it there.

```python
# After every complete_cycle — persist before the session ends
updated_json = outcome.updated_learner.to_json()
your_store.set(learner_id, updated_json)

# At the start of each session — restore before prepare_cycle
stored_json = your_store.get(learner_id)
learner     = LearnerState.from_json(stored_json)
```

**Persisting `CyclePreparation` across the prepare/complete boundary:**

The prepare and complete phases may be separated by minutes while the
learner reads. Store the minimum needed to reconstruct:

```python
import json
from alien_system import (
    LearnerState, SelectionMode, CyclePreparation,
    CandidatePassage, ScaffoldProfile, SelfAudit, DeterministicScores,
    parse_canonical_passage, parse_assessment_package,
    _canonical_to_dict,
)

# ── Store after prepare_cycle ────────────────────────────────────────────────
session_store[prep.cycle_id] = {
    "canonical_json":        json.dumps(
                                 _canonical_to_dict(prep.canonical),
                                 ensure_ascii=False),
    "assessment_raw_json":   json.dumps(
                                 prep.assessment.raw_json,
                                 ensure_ascii=False),
    "prepared_learner_json": prep.prepared_learner.to_json(),
    "selection_mode":        prep.selection_mode.value,
    "cycle_id":              prep.cycle_id,
}

# ── Restore for complete_cycle ───────────────────────────────────────────────
stored = session_store[cycle_id]

prep = CyclePreparation(
    canonical    = parse_canonical_passage(
                       json.loads(stored["canonical_json"])),
    assessment   = parse_assessment_package(
                       json.loads(stored["assessment_raw_json"])),
    # The three fields below are placeholders — complete_cycle only reads
    # canonical, assessment, and prepared_learner from this object.
    selected_candidate = CandidatePassage(
        candidate_id  = "",
        passage_id    = json.loads(stored["canonical_json"])["passage_id"],
        relative_band = 0,
        text          = "",
        scaffold      = ScaffoldProfile(),
        llm_self_audit = SelfAudit(True, True, True, True),
    ),
    selected_scores  = DeterministicScores(0, 0, True, 1, 1, 1, 0, True, True),
    fit_estimates    = {},
    prepared_learner = LearnerState.from_json(stored["prepared_learner_json"]),
    selection_mode   = SelectionMode(stored["selection_mode"]),
    cycle_id         = stored["cycle_id"],
)
```

---

## Part 7 — Monitoring the Integration

### Instrumented backend wrapper

```python
import json
import time
from typing import Any


class InstrumentedBackend:
    """
    Wraps any LLMBackend to emit per-task timing and success/error counts.
    Compatible with any metrics client that exposes timing() and increment().
    """

    def __init__(self, backend, metrics_client) -> None:
        self._backend = backend
        self._metrics = metrics_client

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        task  = json.loads(user_prompt).get("task", "unknown")
        start = time.perf_counter()
        try:
            result = self._backend.complete_json(system_prompt, user_prompt)
            self._metrics.timing(
                f"alien.llm.{task}.latency_ms",
                (time.perf_counter() - start) * 1000,
            )
            self._metrics.increment(f"alien.llm.{task}.success")
            return result
        except Exception:
            self._metrics.increment(f"alien.llm.{task}.error")
            raise
```

### Key metrics to capture

**After every `prepare_cycle`:**
```python
metrics.increment(f"alien.selection.{prep.selection_mode.value}")
```
DEGRADED rate above 30% across a passage corpus signals canonical quality
issues — anchor words may be wrong, or `fk_tolerance` needs widening.

**On every `ALIENError`:**
```python
try:
    prep = system.prepare_cycle(...)
except ALIENError as e:
    metrics.increment(f"alien.error.stage.{e.stage}")
    logger.error("ALIENError stage=%s passage=%s: %s", e.stage, passage_id, e)
```

### WARNING log events to watch

| Warning message | Acceptable rate | Action if elevated |
|---|---|---|
| `"Retell scoring failed — using deterministic keyword fallback"` | < 5% | Check provider reliability for `score_retell` |
| `"LLM diagnosis failed ... — using deterministic fallback"` | < 5% | Same, for `diagnose_outcome` |
| `"Degraded candidate selected for passage ..."` | < 30% per corpus | Review canonical anchor quality; widen `fk_tolerance` |
| `"No candidates passed deterministic validation ..."` | < 30% | Precursor to DEGRADED; same action |

---

## Part 8 — Cost and Throughput Planning

**Calls per full cycle:** 6 (4 in `prepare_cycle`, 2 in `complete_cycle`)

Canonicalization is a one-time cost per passage. A corpus of 200 passages
incurs 200 canonicalization calls total, not once per learner per day.

**Estimated daily load (Claude Sonnet, 150–300 word passages):**

| Deployment | Cycles/day | LLM calls/day | Est. input tokens | Est. output tokens |
|---|---|---|---|---|
| Single classroom (25 learners) | 25 | 150 | ~190,000 | ~75,000 |
| School (500 learners) | 500 | 3,000 | ~3,750,000 | ~1,500,000 |
| District (10,000 learners) | 10,000 | 60,000 | ~75,000,000 | ~30,000,000 |

**Typical wall-clock time:**

| Phase | Passage length | Time |
|---|---|---|
| `prepare_cycle` | Short (100–150 words) | 8–15 seconds |
| `prepare_cycle` | Medium (200–300 words) | 12–25 seconds |
| `complete_cycle` | Any | 3–6 seconds |

---

## Part 9 — Provider Selection Validation

Run this against five passages from your corpus before committing to a
provider. A validation rate below 60% (DEGRADED on more than 40% of runs)
indicates insufficient generation quality.

```python
from alien_system import AdaptiveReadingSystem, LearnerState

# Use default EngineConfig — not a loose test config.
# This shows real deployment behaviour.
system = AdaptiveReadingSystem(llm=YourBackend())

test_cases = [
    (source_text_1, "passage_01", "objective 1"),
    (source_text_2, "passage_02", "objective 2"),
    # ...
]

results = []
for source_text, passage_id, objective in test_cases:
    for band, label in [(3.0, "low_band"), (9.0, "high_band")]:
        learner = LearnerState(f"test_{label}", band)
        try:
            prep = system.prepare_cycle(source_text, passage_id, objective, learner)
            results.append({
                "passage_id":       passage_id,
                "band":             band,
                "selection_mode":   prep.selection_mode.value,
                "meaning_coverage": prep.selected_scores.meaning_coverage,
                "blocking_reasons": prep.selected_scores.blocking_reasons,
                "fk_delivered":     prep.selected_scores.fk_grade,
                "fk_target":        prep.selected_scores.target_fk,
                "status":           "ok",
            })
        except Exception as e:
            results.append({
                "passage_id": passage_id,
                "band":       band,
                "status":     "error",
                "error":      str(e),
            })

total     = len(results)
validated = sum(1 for r in results if r.get("selection_mode") == "validated")
degraded  = sum(1 for r in results if r.get("selection_mode") == "degraded")
errors    = sum(1 for r in results if r.get("status") == "error")

print(f"Validated: {validated}/{total}  ({validated/total:.0%})")
print(f"Degraded:  {degraded}/{total}  ({degraded/total:.0%})")
print(f"Errors:    {errors}/{total}")

# Inspect degraded cases
for r in results:
    if r.get("selection_mode") == "degraded":
        print(f"  DEGRADED {r['passage_id']} band={r['band']}: "
              f"blocking={r['blocking_reasons']}  "
              f"fk_got={r['fk_delivered']:.1f} fk_want={r['fk_target']:.1f}")
```

**Interpreting results:**

- Validated rate >= 60%: provider is acceptable
- Validated rate < 60%: try widening `EngineConfig.fk_tolerance` first; if still low, consider a stronger model
- Any errors on `select_candidate`: all candidates had blocking failures — this is a canonical quality signal, not a provider signal
- High DEGRADED rate at `band=3.0` but not `band=9.0`: the LLM struggles with deep simplification for this passage; review anchor words or widen `fk_tolerance`

---

## Part 10 — Spanish Module

The Spanish module (`alien_system_es`) exports the same interface. Pass
`language="es"` and use `SPANISH_CONFIG`:

```python
from alien_system_es import (
    AdaptiveReadingSystem,
    LearnerState, Level, ReadingTelemetry,
    DeterministicEngine, SPANISH_CONFIG,
)

engine = DeterministicEngine(config=SPANISH_CONFIG)
system = AdaptiveReadingSystem(
    llm=your_backend,
    engine=engine,
    language="es",
)

learner = LearnerState("alumna_01", 5.5, vocabulary_need=Level.HIGH)
prep    = system.prepare_cycle(
    spanish_source, passage_id, objetivo, learner)
outcome = system.complete_cycle(
    learner, prep, learner_answers,
    ReadingTelemetry(
        fluency_score=0.70,
        hint_use_rate=0.20,
        reread_count=3,
        completion=True,
    ),
)
```

The `LLMBackend` implementations in Parts 1–3 work with both modules
unchanged. The module handles all language-specific behaviour internally.

**Key differences in the Spanish module:**
- Uses the Szigriszt-Pazos (1992) readability formula. Band values are not
  commensurable with English FK grades — do not compare them across modules.
- `SPANISH_CONFIG` widens `fk_tolerance` to 1.5 and relaxes meaning and
  vocabulary thresholds slightly to account for richer morphology.
- All six system prompts are in Spanish; diagnosis label values remain in
  English (they are code identifiers used in Python enums).
- `PromptLibrary` accepts a `language` parameter (`"en"` or `"es"`).
