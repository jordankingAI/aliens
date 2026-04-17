"""
alien_dyslexia.py — Dyslexia accommodation extension for ALIEN
---------------------------------------------------------------
A drop-in wrapper around AdaptiveReadingSystem that corrects the signal
interpretation errors the base system makes for learners with dyslexia,
and adds a decoding_barrier diagnosis that separates decoding difficulty
from comprehension difficulty.

USAGE
-----
    from alien_system import LearnerState, Level, ReadingTelemetry, LLMBackend
    from alien_dyslexia import (
        DyslexicLearnerState,
        DyslexicReadingTelemetry,
        DyslexiaAwareSystem,
        DYSLEXIA_ENGINE_CONFIG,
        seed_dyslexic_learner,
    )

    # Seed from a comprehension-level assessment (NOT a decoding score)
    learner = seed_dyslexic_learner(
        learner_id="maya_01",
        comprehension_band=6.5,   # from listening comprehension test
        vocabulary_need=Level.HIGH,
    )

    system = DyslexiaAwareSystem(llm=your_llm_backend)

    prep = system.prepare_cycle(
        source_text, passage_id, objective, learner)

    outcome = system.complete_cycle(
        learner, prep, learner_answers,
        DyslexicReadingTelemetry(
            fluency_score=0.35,      # raw; will be adjusted
            hint_use_rate=0.40,      # raw; will be adjusted
            reread_count=8,
            completion=False,        # timed out; will be adjusted if comprehension ok
            oral_retell_text="Gutenberg invented the press. Books became cheaper.",
        ),
    )

    # outcome.diagnosis will be decoding_barrier, not overloaded
    # outcome.updated_learner.current_band is preserved at 6.5
    # outcome.reading_signals.decoding_adjusted is True
    # outcome.reading_signals.raw_fluency_score preserves the original 0.35

ARCHITECTURE
------------
This module does not modify alien_system.py. It wraps the system at three
precisely defined layers:

  Layer 1 — Extended data types
    DyslexicLearnerState:       adds decoding_disability flag and oral_retell_quality
    DyslexicReadingTelemetry:   adds oral_retell_text for voice-response scoring
    DyslexicReadingSignals:     adds decoding_adjusted flag and raw_* preservation

  Layer 2 — Extended engine
    DyslexiaAwareDeterministicEngine:
      Overrides build_candidate_plan to add decoding_support slot
      Overrides diagnose_fallback to check decoding_barrier before overloaded
      Overrides update_learner_state to apply correct update for decoding_barrier

  Layer 3 — Extended orchestrator
    DyslexiaAwareSystem:
      Overrides build_reading_signals to apply the three signal adjustments
      Overrides score_retell to prefer oral_retell_text when available
      Delegates everything else unchanged to the parent class

SIGNAL ADJUSTMENTS (Layer 3, applied when decoding_disability = True
                    AND comprehension_score >= COMPREHENSION_GUARD)
-----------------------------------------------------------------------
  fluency_score:  effective = max(raw, comprehension * FLUENCY_SCALE)
                  Prevents underchallenged from being blocked by slow decoding
                  when the learner genuinely understood the passage.

  hint_use_rate:  effective = raw * HINT_DISCOUNT
                  Hints for dyslexic learners are predominantly decoding help,
                  not comprehension scaffolding. Halving the rate prevents
                  support_dependent from firing on a decoding-assistance pattern.

  completion:     treat as True when comprehension >= COMPREHENSION_GUARD
                  Timed non-completion reflects decoding speed, not cognitive
                  overload on meaning. The overloaded pathway remains available
                  when comprehension is also low.

All three adjustments are gated on comprehension >= COMPREHENSION_GUARD (0.70).
When comprehension is genuinely low, the unmodified signals drive diagnosis.

OPEN UNCERTAINTIES (documented in alien_dyslexia_guide.docx)
--------------------------------------------------------------
  - FLUENCY_SCALE (0.9) is not empirically validated for ALIEN specifically.
  - HINT_DISCOUNT (0.5) is not empirically validated.
  - COMPREHENSION_GUARD (0.70) may need adjustment for learners with co-occurring
    conditions (ADHD, processing speed differences).
  - LLM reliability on decoding_support passages is unknown without deployment data.
  - Band progression for dyslexic learners may not correlate with standardised
    outcomes in the same way as for non-dyslexic learners.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Sequence

# ── Import everything we need from the base module ───────────────────────────
from alien_system import (
    # Types
    Level, DiagnosisLabel, SelectionMode, ALIENError,
    LearnerState, ReadingTelemetry, ReadingSignals, AssessmentResult,
    CanonicalPassage, CandidatePassage, ScaffoldProfile, SelfAudit,
    CyclePreparation, CycleOutcome, AssessmentPackage,
    DeterministicEngine, EngineConfig, AdaptiveReadingSystem,
    LLMBackend,
    # Helpers
    _json_safe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# Comprehension must be at or above this to apply signal adjustments.
# Below this threshold, unmodified signals are used — the learner may be
# genuinely struggling with meaning, not just decoding.
COMPREHENSION_GUARD: float = 0.70

# fluency_score adjustment: effective = max(raw, comprehension * FLUENCY_SCALE)
# 0.9 ensures that a learner with perfect comprehension gets effective_fluency
# of at least 0.9 — clearing the 0.75 underchallenged threshold.
FLUENCY_SCALE: float = 0.9

# hint_use_rate adjustment: effective = raw * HINT_DISCOUNT
# Halves the hint signal when comprehension is adequate, preventing
# support_dependent from firing on decoding-assistance patterns.
HINT_DISCOUNT: float = 0.5

# raw fluency below this threshold (combined with comprehension >= guard)
# triggers decoding_barrier instead of defaulting to other diagnoses.
DECODING_FLUENCY_THRESHOLD: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Extended data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DyslexicLearnerState:
    """
    Extends LearnerState with two dyslexia-specific fields.

    All standard LearnerState fields are preserved by delegation.
    Use seed_dyslexic_learner() to construct correctly.

    IMPORTANT: seed current_band from a comprehension-level assessment
    (listening comprehension test, or comprehension-only reading measure).
    Do NOT seed from a decoding-level score. Seeding from decoding level is
    the single most damaging error in dyslexia deployment — it locks the
    learner into below-comprehension-level content indefinitely.
    """
    # Core LearnerState fields (mirrored for serialisation convenience)
    learner_id:            str
    current_band:          float
    vocabulary_need:       Level = Level.MEDIUM
    syntax_need:           Level = Level.MEDIUM
    cohesion_need:         Level = Level.MEDIUM
    support_dependence:    Level = Level.MEDIUM
    readiness_to_increase: Level = Level.LOW
    recent_outcomes:       tuple[DiagnosisLabel, ...] = ()
    target_band:           float | None = None
    entry_band:            float | None = None
    cycles_on_passage:     int          = 0

    # Dyslexia-specific fields
    decoding_disability:   bool         = False
    # Set to True when a formal dyslexia diagnosis or specialist assessment exists.
    # Controls signal interpretation and candidate plan throughout the pipeline.
    # Does not change through automated cycles — set by teacher at seeding.

    oral_retell_quality:   float | None = None
    # Retell quality score from an oral or dictated response in the most recent
    # cycle, if collected. Populated by complete_cycle when oral_retell_text is
    # present in DyslexicReadingTelemetry. None means no oral retell was collected.

    def to_base(self) -> LearnerState:
        """Return a standard LearnerState for use with base ALIEN methods."""
        return LearnerState(
            learner_id=self.learner_id,
            current_band=self.current_band,
            vocabulary_need=self.vocabulary_need,
            syntax_need=self.syntax_need,
            cohesion_need=self.cohesion_need,
            support_dependence=self.support_dependence,
            readiness_to_increase=self.readiness_to_increase,
            recent_outcomes=self.recent_outcomes,
            target_band=self.target_band,
            entry_band=self.entry_band,
            cycles_on_passage=self.cycles_on_passage,
        )

    @classmethod
    def from_base(
        cls,
        base: LearnerState,
        decoding_disability: bool = False,
        oral_retell_quality: float | None = None,
    ) -> "DyslexicLearnerState":
        """Promote a standard LearnerState to a DyslexicLearnerState."""
        return cls(
            learner_id=base.learner_id,
            current_band=base.current_band,
            vocabulary_need=base.vocabulary_need,
            syntax_need=base.syntax_need,
            cohesion_need=base.cohesion_need,
            support_dependence=base.support_dependence,
            readiness_to_increase=base.readiness_to_increase,
            recent_outcomes=base.recent_outcomes,
            target_band=base.target_band,
            entry_band=base.entry_band,
            cycles_on_passage=base.cycles_on_passage,
            decoding_disability=decoding_disability,
            oral_retell_quality=oral_retell_quality,
        )

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        # recent_outcomes may contain _DecodingBarrierLabel sentinels which
        # are not standard Enum instances. Serialise them by their .value string.
        d["recent_outcomes"] = [
            (x.value if hasattr(x, "value") else str(x))
            for x in self.recent_outcomes
        ]
        return json.dumps(_json_safe(d))

    @classmethod
    def from_json(cls, s: str) -> "DyslexicLearnerState":
        d = json.loads(s)
        def _load_outcome(v: str):
            if v == DECODING_BARRIER_VALUE:
                return DECODING_BARRIER
            return DiagnosisLabel.from_value(v)
        d["recent_outcomes"] = tuple(_load_outcome(v)
                                     for v in d.get("recent_outcomes", []))
        d["vocabulary_need"]       = Level.from_value(d["vocabulary_need"])
        d["syntax_need"]           = Level.from_value(d["syntax_need"])
        d["cohesion_need"]         = Level.from_value(d["cohesion_need"])
        d["support_dependence"]    = Level.from_value(d["support_dependence"])
        d["readiness_to_increase"] = Level.from_value(d["readiness_to_increase"])
        return cls(**d)

    def to_prompt_dict(self) -> dict[str, Any]:
        """Used when the base class passes learner to LLM prompts."""
        d = self.to_base().to_prompt_dict()
        # Expose decoding_disability to the diagnosis prompt so the LLM
        # can apply the decoding_barrier criterion correctly.
        d["decoding_disability"] = self.decoding_disability
        return d


@dataclass(frozen=True)
class DyslexicReadingTelemetry:
    """
    Extends ReadingTelemetry with an oral retell field.

    oral_retell_text: raw text from a voice-to-text transcription or
    teacher transcription of the learner's oral retell response.
    When present and the learner has decoding_disability=True, score_retell
    uses this text instead of the written Q5 response.
    """
    fluency_score:    float
    hint_use_rate:    float
    reread_count:     int
    completion:       bool
    oral_retell_text: str | None = None

    def to_base(self) -> ReadingTelemetry:
        return ReadingTelemetry(
            fluency_score=self.fluency_score,
            hint_use_rate=self.hint_use_rate,
            reread_count=self.reread_count,
            completion=self.completion,
        )


@dataclass(frozen=True)
class DyslexicReadingSignals:
    """
    Extends ReadingSignals with adjustment audit fields.

    When decoding_adjusted is True, the fluency_score, hint_use_rate,
    and completion values in this object are the ADJUSTED values used
    for diagnosis. The raw values are preserved in raw_fluency_score,
    raw_hint_use_rate, and raw_completion for monitoring and reporting.
    """
    # Standard ReadingSignals fields
    comprehension_score: float
    inference_score:     float
    fluency_score:       float       # adjusted when decoding_adjusted=True
    hint_use_rate:       float       # adjusted when decoding_adjusted=True
    reread_count:        int
    difficulty_rating:   int
    retell_quality:      float
    completion:          bool        # adjusted when decoding_adjusted=True

    # Audit fields
    decoding_adjusted:   bool  = False
    raw_fluency_score:   float = 0.0   # preserved original; 0.0 if not adjusted
    raw_hint_use_rate:   float = 0.0   # preserved original; 0.0 if not adjusted
    raw_completion:      bool  = True  # preserved original; True if not adjusted

    def to_base(self) -> ReadingSignals:
        """Convert to standard ReadingSignals for base-class methods."""
        return ReadingSignals(
            comprehension_score=self.comprehension_score,
            inference_score=self.inference_score,
            fluency_score=self.fluency_score,
            hint_use_rate=self.hint_use_rate,
            reread_count=self.reread_count,
            difficulty_rating=self.difficulty_rating,
            retell_quality=self.retell_quality,
            completion=self.completion,
        )


# ─────────────────────────────────────────────────────────────────────────────
# New diagnosis label
# ─────────────────────────────────────────────────────────────────────────────

# The base DiagnosisLabel enum is frozen (it is imported from alien_system).
# We use a module-level constant string so it can be stored in recent_outcomes
# as a DiagnosisLabel-compatible value and compared consistently.
#
# Implementation choice: rather than subclassing the frozen enum (which is
# fragile) or monkey-patching it (which is wrong), we define a standalone
# string constant and provide a helper that checks for it explicitly.
# The DyslexiaAwareDeterministicEngine returns base DiagnosisLabel values
# for all existing labels and uses a sentinel for decoding_barrier that
# the DyslexiaAwareSystem intercepts before passing to the base update logic.

DECODING_BARRIER_VALUE = "decoding_barrier"

class _DecodingBarrierLabel:
    """
    Sentinel that behaves like a DiagnosisLabel for isinstance checks and
    value comparisons, without modifying the frozen base enum.

    DyslexiaAwareDeterministicEngine returns this from diagnose_fallback
    and update_learner_state when the decoding_barrier condition is met.
    DyslexiaAwareSystem stores it in recent_outcomes as a string and
    converts it back when loading from JSON.
    """
    value = DECODING_BARRIER_VALUE

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _DecodingBarrierLabel):
            return True
        if hasattr(other, "value"):
            return other.value == DECODING_BARRIER_VALUE
        return other == DECODING_BARRIER_VALUE

    def __hash__(self) -> int:
        return hash(DECODING_BARRIER_VALUE)

    def __repr__(self) -> str:
        return f"DiagnosisLabel.DECODING_BARRIER"

    def __str__(self) -> str:
        return DECODING_BARRIER_VALUE


DECODING_BARRIER = _DecodingBarrierLabel()


def is_decoding_barrier(label: Any) -> bool:
    """True if label represents a decoding_barrier diagnosis."""
    if isinstance(label, _DecodingBarrierLabel):
        return True
    if hasattr(label, "value"):
        return label.value == DECODING_BARRIER_VALUE
    return label == DECODING_BARRIER_VALUE


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Extended deterministic engine
# ─────────────────────────────────────────────────────────────────────────────

class DyslexiaAwareDeterministicEngine(DeterministicEngine):
    """
    Extends DeterministicEngine with three dyslexia-specific behaviours:

    1. build_candidate_plan: adds a decoding_support slot when
       learner.decoding_disability is True.

    2. diagnose_fallback: checks decoding_barrier condition BEFORE
       overloaded, preventing timed-out or slow-reading learners from
       being misdiagnosed as overloaded.

    3. update_learner_state: applies the correct state update for
       decoding_barrier (band preserved; support_dependence raised;
       readiness_to_increase lowered).

    All other engine methods delegate unchanged to the parent class.
    """

    def build_candidate_plan(
        self, learner: LearnerState | DyslexicLearnerState
    ) -> list[dict[str, Any]]:
        """
        Standard plan + decoding_support slot for dyslexic learners.

        The decoding_support slot instructs the candidate generator to apply
        surface-level constraints that reduce decoding burden without
        simplifying conceptual content:
          - Maximum 12 words per sentence
          - High-frequency vocabulary throughout (not just required terms)
          - Blank line between every sentence or chunk
          - Pronunciation guides for irregular required vocabulary
          - Active voice, concrete nouns, no nominalisations

        override_length_check=True is set because sentence splitting will
        increase word count, which the length_deviation validator would
        otherwise flag as a warning causing DEGRADED selection.

        CRITICAL: The decoding_support profile does NOT simplify conceptual
        content. The meaning units, causal relationships, and instructional
        vocabulary are preserved at the learner's comprehension band.
        """
        plan = super().build_candidate_plan(learner)

        is_dyslexic = getattr(learner, "decoding_disability", False)
        if is_dyslexic:
            plan.append({
                "relative_band":       0,
                "profile":             "decoding_support",
                "override_length_check": True,
                # Instructs score_candidate to not penalise length deviation
                # for this candidate. Sentence splitting legitimately increases
                # word count — this is intended behaviour, not a quality failure.
            })

        return plan

    def diagnose_fallback(
        self,
        learner: LearnerState | DyslexicLearnerState,
        signals: ReadingSignals | DyslexicReadingSignals,
    ) -> Any:  # DiagnosisLabel | _DecodingBarrierLabel
        """
        Deterministic diagnosis with decoding_barrier checked first.

        Condition for decoding_barrier:
          - learner.decoding_disability is True
          - comprehension_score >= COMPREHENSION_GUARD (learner understood the passage)
          - raw fluency < DECODING_FLUENCY_THRESHOLD (decoding was effortful)

        Uses raw_fluency_score when available (DyslexicReadingSignals) so that
        the decoding_barrier check always runs against the unmodified fluency,
        not the adjusted value. This is correct: decoding_barrier should fire
        when actual decoding speed was low, regardless of what the adjusted
        signal shows the rest of the pipeline.
        """
        is_dyslexic = getattr(learner, "decoding_disability", False)
        if is_dyslexic:
            # Use raw_fluency_score only when signals have already been adjusted
            # (decoding_adjusted=True). When signals are unadjusted, fluency_score
            # IS the raw value. Using raw_fluency_score unconditionally would read
            # its default of 0.0 for unadjusted DyslexicReadingSignals, incorrectly
            # triggering the barrier on any unadjusted signal set.
            is_adjusted = getattr(signals, "decoding_adjusted", False)
            raw_fluency = (
                getattr(signals, "raw_fluency_score", signals.fluency_score)
                if is_adjusted
                else signals.fluency_score
            )
            if (signals.comprehension_score >= COMPREHENSION_GUARD
                    and raw_fluency < DECODING_FLUENCY_THRESHOLD):
                return DECODING_BARRIER

        # All other diagnoses use the standard (already-adjusted) logic.
        return super().diagnose_fallback(learner, signals)

    def update_learner_state(
        self,
        learner:   LearnerState | DyslexicLearnerState,
        diagnosis: Any,
        signals:   ReadingSignals | DyslexicReadingSignals,
    ) -> LearnerState | DyslexicLearnerState:
        """
        Standard update for all existing labels; decoding_barrier rule added.

        decoding_barrier update:
          current_band:          UNCHANGED (learner comprehends at this level)
          vocabulary_need:       unchanged
          syntax_need:           unchanged
          cohesion_need:         unchanged
          support_dependence:    UP (decoding support needed at surface level)
          readiness_to_increase: LOW (hold band until decoding improves)
          cycles_on_passage:     +1

        The decoding_barrier label in recent_outcomes signals to teacher
        dashboards and intervention coordinators that a separate decoding
        intervention track should be activated or intensified.
        """
        if is_decoding_barrier(diagnosis):
            updated_base = replace(
                learner.to_base() if isinstance(learner, DyslexicLearnerState)
                else learner,
                support_dependence=learner.support_dependence.up(),
                readiness_to_increase=Level.LOW,
                cycles_on_passage=learner.cycles_on_passage + 1,
            )
            history = (learner.recent_outcomes + (DECODING_BARRIER,))[
                -self.config.history_limit:]
            updated_base = replace(updated_base, recent_outcomes=history)

            if isinstance(learner, DyslexicLearnerState):
                return DyslexicLearnerState.from_base(
                    updated_base,
                    decoding_disability=learner.decoding_disability,
                    oral_retell_quality=learner.oral_retell_quality,
                )
            return updated_base

        # Standard labels: delegate to parent, then re-wrap if DyslexicLearnerState
        updated_base = super().update_learner_state(
            learner.to_base() if isinstance(learner, DyslexicLearnerState)
            else learner,
            diagnosis,
            signals.to_base() if isinstance(signals, DyslexicReadingSignals)
            else signals,
        )

        if isinstance(learner, DyslexicLearnerState):
            return DyslexicLearnerState.from_base(
                updated_base,
                decoding_disability=learner.decoding_disability,
                oral_retell_quality=learner.oral_retell_quality,
            )
        return updated_base


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Extended orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class DyslexiaAwareSystem(AdaptiveReadingSystem):
    """
    AdaptiveReadingSystem subclass with dyslexia signal adjustments.

    Three methods are overridden:

    build_reading_signals:
        Applies the fluency, hint, and completion adjustments when the
        learner has decoding_disability=True and comprehension is adequate.
        Returns a DyslexicReadingSignals that preserves the raw values for
        monitoring alongside the adjusted values used for diagnosis.

    score_retell:
        Prefers oral_retell_text over the written Q5 response when available
        and the learner has decoding_disability=True.

    complete_cycle:
        Accepts DyslexicLearnerState and DyslexicReadingTelemetry in
        addition to their base types. Passes adjusted signals to diagnosis
        and state update. Preserves oral_retell_quality on the updated learner.

    Everything else — canonicalize_passage, generate_candidates,
    estimate_fit, generate_assessment, score_assessment, prepare_cycle,
    diagnose_outcome, begin_passage_journey — is unchanged and delegates
    to the parent class.
    """

    def __init__(
        self,
        llm:    LLMBackend,
        engine: DyslexiaAwareDeterministicEngine | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            engine=engine or DyslexiaAwareDeterministicEngine(),
            logger=logger,
        )

    def build_reading_signals(
        self,
        assessment_result: AssessmentResult,
        fluency_score:     float,
        hint_use_rate:     float,
        reread_count:      int,
        completion:        bool,
        learner:           LearnerState | DyslexicLearnerState | None = None,
    ) -> DyslexicReadingSignals:
        """
        Construct reading signals, applying dyslexia adjustments when indicated.

        Adjustments fire when ALL of:
          - learner.decoding_disability is True
          - comprehension_score >= COMPREHENSION_GUARD

        The comprehension guard is critical: adjustments must not fire when
        comprehension is genuinely low. A learner who is both struggling to
        decode AND failing to comprehend needs the unmodified signals so the
        standard diagnosis logic can act correctly.

        Returns DyslexicReadingSignals, which carries both the adjusted values
        (used for diagnosis) and the raw values (for monitoring/reporting).
        """
        comprehension = assessment_result.comprehension_score
        is_dyslexic   = (learner is not None
                         and getattr(learner, "decoding_disability", False))
        should_adjust = (is_dyslexic
                         and comprehension >= COMPREHENSION_GUARD)

        if should_adjust:
            eff_fluency    = max(fluency_score, comprehension * FLUENCY_SCALE)
            eff_hint_rate  = hint_use_rate * HINT_DISCOUNT
            eff_completion = True   # non-completion is decoding speed, not overload

            self.logger.debug(
                "Dyslexia signal adjustment for learner %s: "
                "fluency %.2f→%.2f  hints %.2f→%.2f  completion %s→True",
                getattr(learner, "learner_id", "unknown"),
                fluency_score, eff_fluency,
                hint_use_rate, eff_hint_rate,
                completion,
            )
        else:
            eff_fluency    = fluency_score
            eff_hint_rate  = hint_use_rate
            eff_completion = completion

        return DyslexicReadingSignals(
            comprehension_score = comprehension,
            inference_score     = assessment_result.inference_score,
            fluency_score       = round(eff_fluency, 3),
            hint_use_rate       = round(eff_hint_rate, 3),
            reread_count        = int(reread_count),
            difficulty_rating   = int(assessment_result.difficulty_rating),
            retell_quality      = assessment_result.retell_quality,
            completion          = eff_completion,
            # Audit fields
            decoding_adjusted   = should_adjust,
            raw_fluency_score   = round(fluency_score, 3) if should_adjust else 0.0,
            raw_hint_use_rate   = round(hint_use_rate, 3) if should_adjust else 0.0,
            raw_completion      = completion if should_adjust else True,
        )

    def score_retell(
        self,
        canonical:        CanonicalPassage,
        assessment:       AssessmentPackage,
        learner_response: str,
        learner:          LearnerState | DyslexicLearnerState | None = None,
        oral_text:        str | None = None,
    ) -> dict[str, Any]:
        """
        Score retell, preferring oral response for dyslexic learners.

        When oral_text is provided and learner.decoding_disability is True,
        oral_text is scored instead of the written learner_response. The
        written response is not penalised — it simply is not used for scoring.

        This reflects the fundamental principle: for dyslexic learners, a
        sparse written retell does not indicate poor comprehension. The
        written production task is itself a decoding-dependent activity.
        The oral retell provides a cleaner window into comprehension.
        """
        is_dyslexic      = getattr(learner, "decoding_disability", False)
        effective_response = learner_response

        if is_dyslexic and oral_text:
            effective_response = oral_text
            self.logger.debug(
                "Using oral retell for learner %s (decoding_disability=True)",
                getattr(learner, "learner_id", "unknown"),
            )

        return super().score_retell(canonical, assessment, effective_response)

    def complete_cycle(
        self,
        learner:         LearnerState | DyslexicLearnerState,
        prep:            CyclePreparation,
        learner_answers: dict[str, Any],
        telemetry:       ReadingTelemetry | DyslexicReadingTelemetry,
    ) -> CycleOutcome:
        """
        Complete a reading cycle with dyslexia-aware signal processing.

        Differences from the base complete_cycle:
        1. score_retell receives the oral_retell_text from telemetry (if any)
           so it can prefer the oral response for dyslexic learners.
        2. build_reading_signals receives the learner so adjustments can fire.
        3. The diagnosis and state update use the adjusted signals.
        4. If the updated learner is a DyslexicLearnerState, oral_retell_quality
           is updated from the retell score so it is available next cycle.

        The rest of the cycle (assessment scoring, signal construction,
        diagnose_outcome, cycle_id linkage) is unchanged.
        """
        effective_learner = (prep.prepared_learner
                             if prep.prepared_learner is not None
                             else learner)

        # Re-wrap prepared_learner as DyslexicLearnerState if needed
        if (isinstance(learner, DyslexicLearnerState)
                and not isinstance(effective_learner, DyslexicLearnerState)):
            effective_learner = DyslexicLearnerState.from_base(
                effective_learner,
                decoding_disability=learner.decoding_disability,
                oral_retell_quality=learner.oral_retell_quality,
            )

        # Extract oral retell text from telemetry if available
        oral_text = getattr(telemetry, "oral_retell_text", None)

        # Score assessment (MCQ items only — retell handled separately below)
        assessment_result = self.score_assessment(
            canonical=prep.canonical,
            assessment=prep.assessment,
            learner_answers=learner_answers,
        )

        # Score retell — uses oral response if available and appropriate.
        # We re-score Q5 here directly to capture the oral preference.
        # The base score_assessment has already scored it using the written
        # response; we override with the oral score when the learner is dyslexic
        # and oral text is available.
        is_dyslexic = getattr(effective_learner, "decoding_disability", False)
        if is_dyslexic and oral_text:
            retell_item = next(
                (i for i in prep.assessment.items if i.type == "retell_short_response"),
                None,
            )
            if retell_item:
                oral_retell_json = self.score_retell(
                    canonical=prep.canonical,
                    assessment=prep.assessment,
                    learner_response=learner_answers.get("Q5", ""),
                    learner=effective_learner,
                    oral_text=oral_text,
                )
                from alien_system import normalize_retell_score
                oral_rq = normalize_retell_score(
                    int(oral_retell_json.get("raw_score", 0)),
                    int(oral_retell_json.get("max_score", 4)),
                )
                # Rebuild AssessmentResult with the oral retell quality
                from dataclasses import replace as _replace
                assessment_result = _replace(assessment_result,
                    retell_quality=oral_rq,
                    item_scores={**assessment_result.item_scores, "Q5": oral_rq},
                )

        # Build reading signals with dyslexia adjustments
        reading_signals = self.build_reading_signals(
            assessment_result=assessment_result,
            fluency_score=telemetry.fluency_score,
            hint_use_rate=telemetry.hint_use_rate,
            reread_count=telemetry.reread_count,
            completion=telemetry.completion,
            learner=effective_learner,
        )

        # Diagnose using adjusted signals
        # diagnose_outcome calls LLM then falls back to engine.diagnose_fallback.
        # We pass adjusted signals; diagnose_fallback checks raw_fluency_score
        # (preserved in DyslexicReadingSignals) for the decoding_barrier check.
        diagnosis = self.diagnose_outcome(
            effective_learner,
            reading_signals.to_base(),   # LLM prompt uses standard ReadingSignals
        )

        # If LLM returned a label but we know decoding_barrier should fire,
        # override with the deterministic result.
        det_diagnosis = self.engine.diagnose_fallback(effective_learner, reading_signals)
        if is_decoding_barrier(det_diagnosis):
            diagnosis = DECODING_BARRIER

        # Update learner state
        updated = self.engine.update_learner_state(
            effective_learner, diagnosis, reading_signals)

        # Preserve oral_retell_quality on the updated DyslexicLearnerState
        if (isinstance(updated, DyslexicLearnerState)
                and is_dyslexic and oral_text):
            updated = DyslexicLearnerState.from_base(
                updated.to_base(),
                decoding_disability=updated.decoding_disability,
                oral_retell_quality=reading_signals.retell_quality,
            )

        return CycleOutcome(
            diagnosis=diagnosis,
            updated_learner=updated,
            assessment_result=assessment_result,
            reading_signals=reading_signals,
            cycle_id=prep.cycle_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DYSLEXIA_ENGINE_CONFIG = EngineConfig(
    # Band arithmetic — unchanged
    band_step = 0.8,
    min_band  = 0.0,
    max_band  = 12.0,

    # FK tolerance — slightly wider.
    # Decoding_support candidates are longer (sentence splitting) and may
    # have FK grades that differ from the target. The LLM has less precise
    # control over surface form when applying decoding constraints.
    fk_tolerance = 1.5,

    # Meaning thresholds — unchanged. Content preservation requirements
    # are the same for dyslexic learners; only surface form changes.
    unit_sentence_match_threshold = 0.35,
    overall_meaning_threshold     = 0.75,
    vocabulary_threshold          = 0.85,

    # Length thresholds — wider.
    # Sentence splitting (≤12 words per sentence) legitimately increases
    # word count. override_length_check on the plan entry suppresses the
    # warning for decoding_support candidates; these config values govern
    # all other candidates for this learner.
    length_deviation_threshold    = 0.50,
    length_ceiling                = 0.75,

    # Relaxation rates — unchanged
    meaning_relax_per_grade  = 0.02,
    vocab_relax_per_grade    = 0.03,
    length_relax_per_grade   = 0.025,

    # Floors / ceilings — unchanged
    meaning_floor = 0.50,
    vocab_floor   = 0.55,

    # Access / support burden eligibility — unchanged
    min_access_score         = 2,
    max_support_burden_score = 2,

    # Diagnosis thresholds — unchanged.
    # These govern the adjusted signals (which are already corrected for
    # decoding), not the raw signals.
    severe_comprehension_threshold = 0.50,
    low_hint_use_threshold         = 0.10,
    high_hint_use_threshold        = 0.30,

    history_limit = 3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constructor helper
# ─────────────────────────────────────────────────────────────────────────────

def seed_dyslexic_learner(
    learner_id:          str,
    comprehension_band:  float,
    vocabulary_need:     Level = Level.MEDIUM,
    syntax_need:         Level = Level.MEDIUM,
    cohesion_need:       Level = Level.MEDIUM,
) -> DyslexicLearnerState:
    """
    Construct a correctly seeded DyslexicLearnerState.

    comprehension_band MUST come from a comprehension-level assessment:
      - Listening comprehension test (learner hears passage, answers questions)
      - Specialist-administered reading assessment with explicit accommodation
      - Running record with comprehension scoring, not fluency scoring
      - DO NOT use a decoding-level score (e.g., oral reading fluency, word
        reading test). This is the single most common and most damaging error.

    support_dependence is seeded at HIGH (conservative). The system will
    lower it as cycles accumulate underchallenged or well_calibrated diagnoses.

    readiness_to_increase is seeded at LOW. The system will raise it after
    well_calibrated cycles.
    """
    return DyslexicLearnerState(
        learner_id            = learner_id,
        current_band          = comprehension_band,
        vocabulary_need       = vocabulary_need,
        syntax_need           = syntax_need,
        cohesion_need         = cohesion_need,
        support_dependence    = Level.HIGH,   # conservative starting point
        readiness_to_increase = Level.LOW,
        decoding_disability   = True,
    )
