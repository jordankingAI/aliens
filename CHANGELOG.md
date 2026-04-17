# Changelog

All notable changes to ALIENS are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH.

Breaking changes to the `LearnerState` schema, diagnosis labels, or state update rules
are MAJOR version increments. New scaffold dimensions, new language modules, and new
diagnostic signals are MINOR increments. Bug fixes are PATCH increments.

---

## [Unreleased]

*No unreleased changes at this time.*

---

## [1.0.0] — 2026-04-17

Initial public release.

### Modules

- `alien_system.py` — Core English adaptive reading engine (2,349 lines)
- `alien_system_es.py` — Spanish edition using Szigriszt-Pazos readability (2,867 lines)
- `alien_dyslexia.py` — Dyslexia extension module (897 lines)

### Test suites

- `test_alien.py` — 389 tests, English engine
- `test_alien_es.py` — 184 tests, Spanish engine
- `test_alien_dyslexia.py` — 132 tests, dyslexia extension
- `test_fixes.py` — 86 tests, fix regression suite

**Total: 791 tests, all passing.**

### Architecture

- Hybrid deterministic-LLM pipeline: 6 LLM calls per cycle, all hard constraints deterministic.
- `LearnerState`: 11-field frozen dataclass tracking band, five need dimensions, readiness, support dependence, and cycle history.
- Seven diagnostic labels with deterministic fallback: `underchallenged`, `well_calibrated`, `successful_but_support_dependent`, `cohesion_inference_barrier`, `vocabulary_barrier`, `syntax_barrier`, `overloaded`.
- Canonical passage contract: meaning units, sequence constraints, required vocabulary — enforced by the engine as hard blocking rules.
- DEGRADED selection: when no candidate fully passes constraints, the best surface-valid candidate is used with differentiated logging (FK-only DEGRADED vs structural DEGRADED).
- Dyslexia signal adjustments: fluency correction, hint halving, completion override — guarded by a comprehension threshold to prevent false positives.

### Bug fixes included at release (all discovered during end-to-end simulation)

**Fix 1 — Multi-word anchor tokenisation** (`alien_system.py`, `alien_system_es.py`)

`_unit_anchor_tokens()` called `normalize_token()` on each anchor string directly. `normalize_token()` strips spaces before stemming, so `"cottage industries"` became the single concatenated token `"cottageindustri"` — a token that never appears in `content_tokens(sentence)`. Multi-word anchors contributed zero signal to the `anchor_cov` term in `sentence_unit_match_score()`. Fixed by splitting each anchor on whitespace before normalising, matching the behaviour of `content_tokens()` on sentence text.

**Fix 2 — Empty `unit_toks` returned `1.0`** (`alien_system.py`, `alien_system_es.py`)

`sentence_unit_match_score()` returned `1.0` when the MeaningUnit's text produced an empty content-token set (e.g. text consisting only of stopwords or a single short character). Every sentence then scored as a perfect match against every such MU, all MU positions collapsed to sentence 0, and `sequence_ok_from_positions()` returned `False` for every candidate. Fixed by returning `0.0` instead of `1.0` — a MU with no content tokens cannot be reliably located.

**Fix 3 — `ALIENError` module separation** (`alien_system_es.py`)

`alien_system_es.py` defined its own `ALIENError` class independently from `alien_system.py`. A caller using `except alien_system.ALIENError` could not catch errors raised by the Spanish module. Fixed by replacing the standalone class definition with `from alien_system import ALIENError`, making `alien_system_es.ALIENError is alien_system.ALIENError` evaluate to `True`.

**Fix 4 — DEGRADED log did not distinguish FK-only from structural failures** (`alien_system.py`, `alien_system_es.py`)

A single log message covered both cases: all candidates having only FK/length warnings (expected for academic vocabulary) and candidates having blocking failures (meaning coverage, sequence — requiring LLM quality review). These require completely different operator responses. Fixed by checking whether `all_scores` contains any blocking reasons: if none, logs `"DEGRADED (FK/length surface warnings only) — expected for academic domain vocabulary"` with guidance to raise `fk_tolerance`; otherwise logs `"DEGRADED (structural) — review LLM generation quality"`. `EngineConfig.fk_tolerance` comment expanded with domain-specific recommended ranges.

**Fix 5 — `split_sentences` failed at terminal punctuation inside quotation marks** (`alien_system.py`, `alien_system_es.py`)

`_SENT_RE` used the lookbehind `(?<=[.!?])` to find sentence boundaries. When a sentence ended with a period inside a quotation mark — `'Some doors only open from the inside.' Maya read…` — the character before the whitespace was the closing quote `'`, not `.`. The lookbehind failed and the next sentence was merged into the current one, causing both the quotation sentence and the reaction sentence to occupy the same sentence index. This made MU3 and MU4 map to the same position, causing `sequence_ok_from_positions()` to return `False` for every candidate generated from fiction or dialogue passages. Fixed by extending `_SENT_RE` to include a higher-priority alternation `(?<=[.!?]['"'\u2019\u201d])\s+` covering period/!/? followed by a closing quotation character.

---

## Version history format

Future entries will follow this structure:

```
## [X.Y.Z] — YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing behaviour (with migration guidance if breaking)

### Fixed
- Bug fixes (root cause, not just symptom)

### Deprecated
- Features that will be removed in a future major version

### Removed
- Features removed in this version

### Security
- Security fixes (reference SECURITY.md advisory number)
```
