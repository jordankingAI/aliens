from __future__ import annotations

"""
ALIEN — Adaptive Literacy Instruction Engagement Network (edición en español)
------------------------------------------------------------------------------
A hybrid instructional system that:
1) canonicalizes a source passage into protected meaning units,
2) asks an LLM to generate semantically aligned passage variants,
3) validates and selects the best variant deterministically,
4) asks an LLM to generate a diagnostic comprehension check,
5) converts learner performance into reading signals,
6) diagnoses the outcome and updates learner state.

Self-contained: standard library only (+ logging).
Wire your LLM by implementing LLMBackend.complete_json(system, user) -> dict.

Key architectural invariants:
- Every candidate is scored exactly once per cycle (no repeated scoring).
- The deterministic layer is the final authority on hard constraints,
  candidate selection, and learner-state updates.
- LLM failures in diagnosis are logged and fall back deterministically.
- Validation thresholds scale with passage-distance so distant rewrites
  are not rejected for being different from a hard-to-match source.

Language support:
  This module defaults to Spanish ("es"). English ("en") is also available
  by passing language="en" to AdaptiveReadingSystem().

  The Spanish configuration:
  - Uses the Szigriszt-Pazos (1992) Perspicuity Index instead of
    Flesch-Kincaid. The source_fk field on CanonicalPassage stores an
    IFSZ-derived grade equivalent (0-12), not an FK grade.
  - Uses a Spanish rule-based morphological stemmer (Snowball-style).
  - Uses Spanish stopwords and negation markers.
  - Uses Spanish-language LLM system prompts throughout the pipeline.

  Use SPANISH_CONFIG (defined at module bottom) as the EngineConfig for
  Spanish deployments. Band values are NOT commensurable between the
  English and Spanish configurations — they use different underlying scales.
"""

import dataclasses
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, Sequence


# ============================================================
# Enums
# ============================================================

class Level(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

    @property
    def score(self) -> int:
        return {"low": 1, "medium": 2, "high": 3}[self.value]

    @classmethod
    def from_value(cls, value: str) -> "Level":
        v = value.strip().lower()
        mapping = {"low": cls.LOW, "medium": cls.MEDIUM, "high": cls.HIGH}
        if v not in mapping:
            raise ValueError(f"Invalid level: {value!r}")
        return mapping[v]

    @classmethod
    def from_score(cls, score: int) -> "Level":
        return {1: cls.LOW, 2: cls.MEDIUM, 3: cls.HIGH}[max(1, min(3, score))]

    def up(self) -> "Level":
        return Level.from_score(self.score + 1)

    def down(self) -> "Level":
        return Level.from_score(self.score - 1)


class DiagnosisLabel(Enum):
    UNDERCHALLENGED               = "underchallenged"
    WELL_CALIBRATED               = "well_calibrated"
    SUCCESSFUL_BUT_SUPPORT_DEPENDENT = "successful_but_support_dependent"
    VOCABULARY_BARRIER            = "vocabulary_barrier"
    SYNTAX_BARRIER                = "syntax_barrier"
    COHESION_INFERENCE_BARRIER    = "cohesion_inference_barrier"
    OVERLOADED                    = "overloaded"

    @classmethod
    def from_value(cls, value: str) -> "DiagnosisLabel":
        v = value.strip().lower()
        for item in cls:
            if item.value == v:
                return item
        raise ValueError(f"Invalid diagnosis label: {value!r}")


class SelectionMode(Enum):
    """
    Records how a candidate was chosen.
    VALIDATED: candidate passed all deterministic constraints.
    DEGRADED:  no candidate fully passed; best available selected on
               non-semantic criteria (FK/length failures only).
    """
    VALIDATED = "validated"
    DEGRADED  = "degraded"


# ALIENError is imported from alien_system so that callers routing across both
# the English and Spanish modules can catch errors from either with a single
# except clause.  After this change:
#
#   alien_system_es.ALIENError is alien_system.ALIENError  → True
#
# Without this, a caller that does:
#
#   from alien_system import ALIENError
#   try:
#       es_system.complete_cycle(...)
#   except ALIENError:          # ← silently misses alien_system_es errors
#       ...
#
# would need to also import alien_system_es.ALIENError and list both in the
# except clause — a subtle, hard-to-detect omission in language-routing code.
#
# ValidationError in this module subclasses the imported ALIENError, so it
# is also caught by a single except ALIENError handler regardless of which
# module raised it.
from alien_system import ALIENError


# ============================================================
# Text utilities
# ============================================================

_WORD_RE     = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+")
# _SENT_RE splits on sentence boundaries.
# The extended alternation handles four cases in priority order:
#   (?<=[.!?]['"'\u2019\u201d])\s+  — terminal punct INSIDE a closing quote, then whitespace.
#                                      Frequent in fiction and quoted speech: without this,
#                                      the closing quote defeats the plain lookbehind.
#                                      p.ej. «Sólo se abren desde dentro.» María → dos oraciones.
#   (?<=[.!?])\s+                   — puntuación terminal seguida de espacio (caso estándar).
#   (?<=[.!?])\n+                   — puntuación terminal seguida de saltos de línea.
#   \n{2,}                          — línea en blanco (salto de párrafo).
_SENT_RE     = re.compile(
    r"(?<=[.!?]['\"\u2019\u201d])\s+"
    r"|(?<=[.!?])\s+"
    r"|(?<=[.!?])\n+"
    r"|\n{2,}"
)
_SENT_ROUGH  = re.compile(r"[.!?]\s+")
_NEGATION    = {
    "no",       # negación primaria; la más frecuente
    "nunca",    # never
    "jamás",    # never (más enfático / literario)
    "nada",     # nothing
    "nadie",    # nobody
    "ningún",   # ninguno (masc. ante sustantivo)
    "ninguno",  # ninguno (masc. aislado)
    "ninguna",  # ninguna (fem.)
    "tampoco",  # neither/nor (negativo de también)
    "ni",       # neither/nor (correlativo, p.ej. ni X ni Y)
    "sin",      # without
    "apenas",   # hardly/scarcely (negación escalar débil)
}
_STOPWORDS   = {
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Preposiciones y contracciones
    "a", "al", "ante", "bajo", "con", "contra", "de", "del",
    "desde", "durante", "en", "entre", "hacia", "hasta", "mediante",
    "para", "por", "según", "sin", "sobre", "tras",
    # Conjunciones y conectores
    "y", "e", "o", "u", "pero", "sino", "aunque", "porque", "pues",
    "como", "cuando", "donde", "mientras", "que", "si", "ya",
    "ni", "más", "aún", "también", "además", "sin", "embargo",
    # Demostrativos
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    # Pronombres personales
    "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "usted", "ustedes",
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
    # Posesivos
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
    # Cópula y auxiliares
    "es", "son", "era", "eran", "fue", "fueron", "ser", "sido",
    "está", "están", "estaba", "estaban", "estuvo", "estuvieron", "estar",
    "ha", "han", "había", "habían", "hubo", "hubieron", "haber",
    "tiene", "tienen", "tenía", "tenían", "tuvo", "tuvieron",
    # Adverbios de alta frecuencia
    "muy", "más", "menos", "bien", "mal", "así", "tan", "tanto",
    "aquí", "ahí", "allí", "antes", "después", "ahora", "entonces",
    "no", "sí", "solo", "sólo",
    # Relativos e interrogativos
    "que", "quien", "quienes", "cual", "cuales", "cuyo", "cuya",
    "cuándo", "cómo", "dónde", "qué", "quién", "cuál",
    # Indefinidos y cuantificadores
    "todo", "toda", "todos", "todas", "cada", "otro", "otra",
    "otros", "otras", "algún", "alguno", "alguna", "algunos", "algunas",
    "mismo", "misma", "mismos", "mismas",
}


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Split on punctuation boundaries; fall back to rough split for unpunctuated text."""
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(parts) <= 1 and len(text.split()) > 12:
        rough = [s.strip() for s in _SENT_ROUGH.split(text) if s.strip()]
        if len(rough) > 1:
            return rough
    return parts or [text.strip()]


def normalize_token(token: str) -> str:
    """
    Spanish morphological normaliser (Snowball-style, inline).

    Applies a practical stemming sequence targeting the most common verb and
    noun inflections in educational Spanish text:

    Verb suffixes handled:
      Present/past participles: -ando, -iendo, -ándose, -iéndose
      Past participles:         -ado/-ada/-ados/-adas, -ido/-ida/-idos/-idas
      Imperfect indicative:     -aba/-abas/-aban, -ía/-ías/-ían
      Preterite:                -ó, -aron, -ió, -ieron
      Infinitives:              -ar, -er, -ir, -arse, -erse, -irse

    Nominal suffixes handled:
      Plural -es (papeles → papel)
      Plural -s  (libros → libro)

    Does NOT strip gender endings (-o/-a) — these carry semantic content in
    Spanish and stripping them creates false matches (banco ≠ banca).

    Symmetric: the same rules fire on both the anchor text and the candidate
    text, so even imperfect stems compare consistently across both sides.

    Upgrade path: inject a SpaCy-backed tokenizer via DeterministicEngine's
    optional tokenizer parameter for production-quality lemmatization.
    """
    # Normalise: lowercase, keep accented letters and apostrophe
    token = re.sub(r"[^a-záéíóúüñ']", "", token.lower())
    if len(token) <= 3:
        return token

    # ── Present/past participle (reflexive forms first, longest suffix first) ──
    for sfx in ("ándose", "iéndose", "ando", "iendo"):
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]

    # ── Past participle ───────────────────────────────────────────────────────
    for sfx in ("ados", "idas", "idos", "adas", "ado", "ada", "ido", "ida"):
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]

    # ── Imperfect indicative ──────────────────────────────────────────────────
    for sfx in ("aban", "abas", "aba", "ían", "ías", "ía"):
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]

    # ── Preterite ─────────────────────────────────────────────────────────────
    for sfx in ("ieron", "aron", "ió", "ó"):
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]

    # ── Infinitives (reflexive forms first) ───────────────────────────────────
    for sfx in ("arse", "erse", "irse", "ar", "er", "ir"):
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]

    # ── Plural -es (papeles → papel, árboles → árbol) ─────────────────────────
    if token.endswith("es") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 3:
            return stem

    # ── Plural -s (libros → libro, casas → casa) ──────────────────────────────
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        stem = token[:-1]
        if len(stem) >= 3:
            return stem

    return token


def content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in words(text):
        norm = normalize_token(tok)
        if norm and norm not in _STOPWORDS and len(norm) > 2:
            out.add(norm)
    return out


def has_negation(text: str) -> bool:
    return any(m in set(words(text)) for m in _NEGATION)


def ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator <= 0 else numerator / denominator


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Spanish vowel sets for syllable counting ────────────────────────────────
_SPANISH_STRONG_VOWELS = set("aeoáéó")
_SPANISH_WEAK_VOWELS   = set("iuíúü")


def count_syllables(word: str) -> int:
    """
    Spanish syllable counter (Szigriszt-Pazos / educational-text calibration).

    Counts vowel nuclei using simplified Spanish phonotactics:
    - Each strong vowel (a, e, o + accented forms) always forms its own nucleus.
    - A weak vowel (i, u + accented/diaeresis forms) adjacent to a strong vowel
      forms a diphthong (one nucleus). Two consecutive weak vowels form hiatus
      (two nuclei), e.g. ciudad → ciu(1) + dad(1) = 2.
    - Terminal -y treated as a weak vowel (hay, rey, hoy → 1 syllable).
    - No silent-e rule: Spanish terminal e is always phonetically realised.

    Symmetric: consistent across both anchor text and candidate text, so even
    approximate stems compare correctly in the meaning matcher.
    """
    word = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    if not word:
        return 0
    # Treat terminal y as weak vowel
    if word.endswith("y"):
        word = word[:-1] + "i"
    if not word:
        return 1

    syllables = 0
    prev_type = None  # None | "strong" | "weak"
    for ch in word:
        if ch in _SPANISH_STRONG_VOWELS:
            syllables += 1
            prev_type = "strong"
        elif ch in _SPANISH_WEAK_VOWELS:
            if prev_type == "weak":
                # Two consecutive weak vowels = hiatus = two separate nuclei
                syllables += 1
            elif prev_type is None:
                # Weak vowel after consonant (or word start) = new nucleus
                syllables += 1
            # else: weak after strong = diphthong = no new syllable
            prev_type = "weak"
        else:
            prev_type = None  # consonant resets diphthong context
    return max(1, syllables)


def readability_grade(text: str) -> float:
    """
    Compute a readability band score for Spanish text using the
    Szigriszt-Pazos (1992) Perspicuity Index (IFSZ), mapped to a 0-12
    grade equivalent compatible with EngineConfig band arithmetic.

    Formula:
        IFSZ = 206.835 - (62.3 × syllables/words) - (words/sentences)

    The IFSZ runs 0-100 with higher = easier (opposite of FK grade).
    Mapping to a 0-12 ascending-difficulty band:
        grade = (100 - IFSZ) × 0.15   (clamped to [0.0, 12.0])

    This maps:
        IFSZ 100 (trivially simple) → band 0.0
        IFSZ ~33 (very difficult)   → band ~10.0

    NOTE: The returned value is an IFSZ-derived grade equivalent, not an FK
    grade. The field is named source_fk throughout the codebase for API
    compatibility. Band values from this function are NOT commensurable with
    FK grades from the English alien_system.py.
    """
    toks = words(text)
    if not toks:
        return 0.0
    sent_count = max(1, len(split_sentences(text)))
    syllables  = sum(count_syllables(t) for t in toks)
    asw        = syllables / len(toks)      # avg syllables per word
    asl        = len(toks) / sent_count     # avg sentence length (words)
    ifsz       = 206.835 - (62.3 * asw) - asl
    ifsz       = max(0.0, min(100.0, ifsz))
    grade      = (100.0 - ifsz) * 0.15
    return round(max(0.0, min(12.0, grade)), 2)


# Alias — all internal callers use flesch_kincaid_grade; the alias keeps
# CanonicalPassage.__post_init__, DeterministicEngine, and existing tests
# working without modification. Rename to readability_grade in a future refactor.
flesch_kincaid_grade = readability_grade


# ============================================================
# Data contracts
# ============================================================

@dataclass(frozen=True)
class MeaningUnit:
    id:       str
    text:     str
    required: bool           = True
    anchors:  tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceConstraint:
    before: str
    after:  str


@dataclass(frozen=True)
class VocabularyTerm:
    term:          str
    required:      bool = True
    gloss_allowed: bool = True


@dataclass(frozen=True)
class CanonicalPassage:
    passage_id:             str
    source_text:            str
    instructional_objective: str
    meaning_units:          tuple[MeaningUnit, ...]
    sequence_constraints:   tuple[SequenceConstraint, ...] = ()
    must_preserve_vocabulary: tuple[VocabularyTerm, ...]  = ()
    source_fk:              float = 0.0   # computed deterministically at build time

    def __post_init__(self) -> None:
        # Compute source_fk if not provided (covers both LLM-parsed and hand-built instances)
        if self.source_fk == 0.0 and self.source_text:
            object.__setattr__(self, "source_fk", flesch_kincaid_grade(self.source_text))


@dataclass(frozen=True)
class SelfAudit:
    meaning_preserved:    bool
    sequence_preserved:   bool
    objective_preserved:  bool
    same_passage_identity: bool
    notes:                str = ""
    # Optional structured coverage map: {mu_id: bool} from llm_self_audit.
    # When present, used in score_candidate as a fast-path MU check
    # before running the expensive lexical meaning_profile computation.
    meaning_unit_coverage: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ScaffoldProfile:
    vocabulary_support: Level = Level.LOW
    syntax_support:     Level = Level.LOW
    cohesion_support:   Level = Level.LOW
    chunking_support:   Level = Level.LOW
    inference_support:  Level = Level.LOW

    def total_support(self) -> int:
        return sum(lvl.score for lvl in (
            self.vocabulary_support, self.syntax_support,
            self.cohesion_support,   self.chunking_support,
            self.inference_support,
        ))

    def to_dict(self) -> dict[str, str]:
        return {
            "vocabulary_support": self.vocabulary_support.value,
            "syntax_support":     self.syntax_support.value,
            "cohesion_support":   self.cohesion_support.value,
            "chunking_support":   self.chunking_support.value,
            "inference_support":  self.inference_support.value,
        }


@dataclass(frozen=True)
class CandidatePassage:
    candidate_id:   str
    passage_id:     str
    relative_band:  int
    text:           str
    scaffold:       ScaffoldProfile
    llm_self_audit: SelfAudit


@dataclass(frozen=True)
class DeterministicScores:
    fk_grade:            float
    target_fk:           float
    fk_within_tolerance: bool
    meaning_coverage:    float
    avg_meaning_score:   float
    vocabulary_coverage: float
    length_deviation:    float
    sequence_ok:         bool
    passed_constraints:  bool
    # blocking_reasons: disqualify entirely (self-audit, meaning, sequence)
    # warning_flags: surface failures allowing degraded selection (FK, length)
    blocking_reasons:    tuple[str, ...] = ()
    warning_flags:       tuple[str, ...] = ()
    selection_mode:      SelectionMode   = SelectionMode.VALIDATED

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """Backwards-compatible union of blocking_reasons + warning_flags."""
        return self.blocking_reasons + self.warning_flags


@dataclass(frozen=True)
class FitEstimate:
    access:         Level
    growth:         Level
    support_burden: Level
    reason:         str = ""

    @property
    def utility(self) -> int:
        return 2 * self.access.score + 2 * self.growth.score - self.support_burden.score


@dataclass(frozen=True)
class ReadingSignals:
    comprehension_score: float
    inference_score:     float
    fluency_score:       float
    hint_use_rate:       float
    reread_count:        int
    difficulty_rating:   int    # learner perceived difficulty Q6 (1=easy, 5=hard)
    retell_quality:      float
    completion:          bool


@dataclass(frozen=True)
class ReadingTelemetry:
    """Behavioral signals collected during the reading session."""
    fluency_score:  float
    hint_use_rate:  float
    reread_count:   int
    completion:     bool


@dataclass(frozen=True)
class LearnerState:
    learner_id:            str
    current_band:          float
    vocabulary_need:       Level = Level.MEDIUM
    syntax_need:           Level = Level.MEDIUM
    cohesion_need:         Level = Level.MEDIUM
    support_dependence:    Level = Level.MEDIUM
    readiness_to_increase: Level = Level.LOW
    recent_outcomes:       tuple[DiagnosisLabel, ...] = ()
    # Journey tracking (optional; set when a passage is first assigned)
    target_band:           float | None = None   # source_fk of current passage
    entry_band:            float | None = None   # band when passage was first assigned
    cycles_on_passage:     int          = 0

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "learner_id":            self.learner_id,
            "current_band":          self.current_band,
            "vocabulary_need":       self.vocabulary_need.value,
            "syntax_need":           self.syntax_need.value,
            "cohesion_need":         self.cohesion_need.value,
            "support_dependence":    self.support_dependence.value,
            "readiness_to_increase": self.readiness_to_increase.value,
            "recent_outcomes":       [x.value for x in self.recent_outcomes],
            "target_band":           self.target_band,
            "cycles_on_passage":     self.cycles_on_passage,
        }

    def to_json(self) -> str:
        """
        Serialize to JSON. Uses _json_safe so any future enum fields are
        handled automatically — no manual per-field overrides needed.
        """
        return json.dumps(_json_safe(dataclasses.asdict(self)))

    @classmethod
    def from_json(cls, s: str) -> "LearnerState":
        d = json.loads(s)
        d["recent_outcomes"]       = tuple(DiagnosisLabel.from_value(v) for v in d.get("recent_outcomes", []))
        d["vocabulary_need"]       = Level.from_value(d["vocabulary_need"])
        d["syntax_need"]           = Level.from_value(d["syntax_need"])
        d["cohesion_need"]         = Level.from_value(d["cohesion_need"])
        d["support_dependence"]    = Level.from_value(d["support_dependence"])
        d["readiness_to_increase"] = Level.from_value(d["readiness_to_increase"])
        return cls(**d)


@dataclass(frozen=True)
class AssessmentItem:
    id:      str
    type:    str
    target:  Any
    payload: dict[str, Any]


@dataclass(frozen=True)
class AssessmentPackage:
    assessment_blueprint: dict[str, Any]
    items:                tuple[AssessmentItem, ...]
    scoring_blueprint:    dict[str, Any]
    signal_mapping:       dict[str, Any]
    raw_json:             dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssessmentResult:
    item_scores:         dict[str, float]
    comprehension_score: float
    inference_score:     float
    vocabulary_score:    float
    retell_quality:      float
    difficulty_rating:   int    # from Q6 self-rating


@dataclass(frozen=True)
class CyclePreparation:
    canonical:          CanonicalPassage
    selected_candidate: CandidatePassage
    selected_scores:    DeterministicScores
    assessment:         AssessmentPackage
    fit_estimates:      dict[str, FitEstimate]
    all_scores:         dict[str, DeterministicScores] = field(default_factory=dict)
    selection_mode:     SelectionMode = SelectionMode.VALIDATED
    # Journey-initialised learner snapshot: target_band, entry_band, and
    # cycles_on_passage are set by begin_passage_journey() during prepare_cycle().
    # complete_cycle() uses this snapshot so journey fields are never lost.
    prepared_learner:   LearnerState | None = None
    cycle_id:           str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          float = field(default_factory=time.time)


@dataclass(frozen=True)
class CycleOutcome:
    diagnosis:         DiagnosisLabel
    updated_learner:   LearnerState
    assessment_result: AssessmentResult
    reading_signals:   ReadingSignals
    cycle_id:          str   = ""
    timestamp:         float = field(default_factory=time.time)


@dataclass(frozen=True)
class EngineConfig:
    band_step: float = 0.8
    min_band:  float = 0.0
    max_band:  float = 12.0

    # FK tolerance (tolerancia de legibilidad): cuántos puntos del índice
    # Szigriszt-Pazos puede desviarse un candidato de target_fk antes de recibir
    # una advertencia fk_out_of_tolerance.  El índice de legibilidad es una medida
    # superficial — es una advertencia, no una restricción bloqueante.  Los
    # candidatos que superan la tolerancia siguen siendo elegibles para selección
    # DEGRADED.
    #
    # El valor predeterminado 1.2 es apropiado para textos generales/narrativos
    # donde es factible reescribir cerca del índice objetivo.
    #
    # Para textos académicos densos (historia, ciencia, derecho) donde el
    # vocabulario requerido es inherentemente polisílabo (catalysó, legislación,
    # soberanía, revolucionario), las reescrituras para alumnos de banda baja
    # puntuarán consistentemente por encima de la banda objetivo aunque la
    # estructura de las oraciones sea sencilla.  Esto es una propiedad del
    # vocabulario del dominio, no un fallo de generación.
    # Valores de fk_tolerance recomendados por dominio:
    #   Contenido general/narrativo  : 1.2 – 2.0  (valor predeterminado)
    #   Contenido académico/técnico  : 2.5 – 4.0  (aumentar para reducir DEGRADED)
    #   Contenido muy especializado  : 4.0 – 6.0  (tratar FK como indicativo)
    fk_tolerance:              float = 1.2

    unit_sentence_match_threshold: float = 0.35   # per-sentence match threshold
    overall_meaning_threshold:     float = 0.75   # base; scales down with distance
    vocabulary_threshold:          float = 0.85   # base; scales down with distance
    length_deviation_threshold:    float = 0.40   # base; scales up with distance

    # How much each threshold relaxes per FK grade of passage distance.
    # Capped so thresholds never drop below a minimum floor.
    meaning_relax_per_grade:   float = 0.02
    vocab_relax_per_grade:     float = 0.03
    length_relax_per_grade:    float = 0.025
    meaning_floor:             float = 0.50
    vocab_floor:               float = 0.55
    length_ceiling:            float = 0.70

    min_access_score:      int   = 2
    max_support_burden_score: int = 2

    severe_comprehension_threshold: float = 0.50
    low_hint_use_threshold:         float = 0.10
    high_hint_use_threshold:        float = 0.30

    history_limit: int = 3


# ============================================================
# LLM backend protocol
# ============================================================

class LLMBackend(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return parsed JSON matching the requested contract."""
        ...


class TaskRoutingMockLLM:
    """
    Development stub that routes by 'task' key in the JSON user prompt.
    Supports error injection and records all calls for inspection.
    Replaces the old MockLLMBackend whose substring matching was fragile.
    """
    def __init__(
        self,
        responses:      dict[str, dict[str, Any]] | None = None,
        error_on_tasks: set[str] | None = None,
    ) -> None:
        self.responses      = responses      or {}
        self.error_on_tasks = error_on_tasks or set()
        self.call_log:      list[tuple[str, str]] = []   # (task, user_prompt)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        data = json.loads(user_prompt)
        task = data.get("task", "")
        self.call_log.append((task, user_prompt))
        if task in self.error_on_tasks:
            raise RuntimeError(f"Injected error for task: {task!r}")
        if task in self.responses:
            return self.responses[task]
        raise NotImplementedError(
            f"TaskRoutingMockLLM has no response for task {task!r}. "
            f"Available: {sorted(self.responses)}"
        )


# ============================================================
# Prompt library
# ============================================================

class PromptLibrary:
    """
    Holds the six LLM system prompts (in English and Spanish) and the six
    user prompt builders (language-neutral JSON constructors).

    language: "en" (English) or "es" (Spanish, default for this module).
    """

    SUPPORTED_LANGUAGES = ("en", "es")

    # ── English system prompts ───────────────────────────────────────────────

    _CANONICALIZER_EN = """
You are a canonical passage extractor inside an adaptive reading system.

Your task is to convert a source passage into a protected instructional representation.
You must identify:
- protected meaning units,
- protected sequence constraints,
- required vocabulary,
- instructional objective.

Rules:
- Extract core events, claims, or ideas only.
- Do not include trivial details unless they are instructionally essential.
- Meaning units must be concise and explicit.
- Anchors must be words that appear in the SOURCE TEXT — not paraphrases.
  The validator uses them to match rewrites, so they must be lexically present.
- Sequence constraints should only be included when order matters causally or logically.
- Required vocabulary should include lesson-critical terms only.
- Do not split a single event into multiple meaning units unless independent
  assessment of those parts is instructionally necessary. Overfragmentation
  makes validation brittle and retell scoring unreliable.
- Output valid JSON only.
""".strip()

    _CANDIDATE_GENERATOR_EN = """
You are a constrained passage rewriter inside an adaptive reading system.

Your task is to generate semantically aligned versions of the same passage.
You must preserve:
- all required meaning units (every one — omitting any is a failure),
- the instructional objective,
- required vocabulary (use inline glosses when gloss_allowed is true),
- passage identity.

You may vary:
- sentence length and structure,
- vocabulary burden (replace Tier 3 with Tier 2 words),
- syntax complexity,
- cohesion support (transitional phrases),
- chunking,
- inference support.

LEARNER-STATE REWRITING GUIDANCE:
Apply these rules based on the learner_state fields:

vocabulary_need HIGH
  Replace low-frequency / Tier 3 vocabulary with simpler equivalents.
  Retain required vocabulary terms; add a parenthetical definition when gloss_allowed is true,
  e.g. "manuscripts (books written by hand)".

syntax_need HIGH
  Prefer active voice. Break embedded clauses into separate sentences.
  Move clausal modifiers after the main predicate.
  Avoid participial phrases at the start of sentences.

cohesion_need HIGH
  Add explicit transitional phrases: "First, ...", "Because of this, ...",
  "As a result, ...", "This meant that ...".
  Make causal relationships explicit rather than implied.

support_dependence HIGH / readiness_to_increase LOW
  Increase chunking (shorter paragraphs). Add explicit topic sentences.
  Scaffold inference: state conclusions the reader might otherwise need to infer.

readiness_to_increase HIGH / support_dependence LOW
  Allow more complex syntax. Reduce glosses. Preserve some Tier 2 vocabulary.
  Let the reader make reasonable inferences without prompting.

Rules:
- Do not delete any required meaning unit. Every MU must appear in every candidate.
- Do not change the central claim or event structure.
- Do not reverse required sequence constraints.
- Generate exactly the candidates in the candidate_plan — one per plan entry.
- In llm_self_audit, include a "meaning_unit_coverage" object listing each MU id
  and whether it is present in this candidate, e.g.:
  "meaning_unit_coverage": {"MU1": true, "MU2": true, "MU3": false}
- Output valid JSON only.
""".strip()

    _FIT_ESTIMATOR_EN = """
You are a learner-text fit estimator inside an adaptive reading system.

Your task is to rate each candidate for:
- access:         how likely the learner can understand the passage adequately
- growth:         how likely the passage provides meaningful productive challenge
- support_burden: how much the passage relies on scaffolding to be successful

Labels: low | medium | high  (no other values)

Rules:
- Rate each candidate separately based on learner_state and deterministic_scores.
- A passage well above learner band with no glosses = access:medium or low.
- A passage well below learner band = growth:low, even with access:high.
- High scaffold (vocabulary/cohesion/chunking support) raises support_burden.
- Do not prefer easier candidates merely because they maximize access.
  Favor candidates that preserve productive challenge when access remains adequate.
- Keep reason concise (1–2 sentences).
- Output valid JSON only.
""".strip()

    _ASSESSMENT_GENERATOR_EN = """
You are an assessment generator inside an adaptive reading system.

Your job is to create a short diagnostic comprehension check for a learner who has just read a passage.

Generate from the canonical passage model first; use the delivered passage only to calibrate wording.
Your purpose is to produce the exact signals needed for the system to decide how to adapt the next passage.

Generate EXACTLY these six items in this order:
  Q1  literal_mcq          — tests a single stated fact (target a required meaning unit)
  Q2  sequence_mcq         — tests temporal or causal ordering of events
                           target must be {"meaning_unit_ids": ["MUx", "MUy"], "relation": "before"}
  Q3  inference_mcq        — tests a cause-effect or implication NOT directly stated
  Q4  vocabulary_mcq       — tests a required vocabulary term in context
  Q5  retell_short_response — open-ended gist retell scored against canonical MUs
  Q6  self_rating          — learner's perceived difficulty (scale 1–5)

Each MCQ item MUST have exactly these fields:
  id, type, target, question, choices (array of {id, text}), correct_answer

Each retell item MUST have:
  id, type, target (null), prompt, rubric ({max_score, criteria: [{points, description}]})
  rubric criteria must map exactly to required meaning units.

Each self_rating item MUST have:
  id, type, target (null), prompt, scale ("1-5")

scoring_blueprint MUST include all four keys:
  literal_item_ids, sequence_item_ids, inference_item_ids, vocabulary_item_ids

signal_mapping MUST include:
  comprehension_score: weights covering all four MCQ items (sum to 1.0)
  inference_score: weights covering Q3 and Q5 (sum to 1.0)
  vocabulary_signal, retell_quality, difficulty_signal

Rules:
- Distractors must be plausible and diagnostic (each distractor reveals a specific misconception).
- Do not create trick questions.
- The retell rubric must be scoreable: each criterion must include a
  "meaning_unit_ids" list, e.g. {"points":1,"meaning_unit_ids":["MU1"],"description":"..."}
- Rubric max_score must equal the sum of all criteria points.
- Output valid JSON only.
""".strip()

    _RETELL_SCORER_EN = """
You are a retell scorer inside an adaptive reading system.

Your task is to score a learner's short retell against the provided rubric.
Score ONLY according to rubric criteria — do not award credit for content not in the rubric.
Return:
- raw_score         (integer)
- max_score         (integer — echo the rubric max_score)
- matched_meaning_units  (list of MU ids that were covered)
- matched_relationships  (list of causal/temporal relationships identified)
- concise_reason    (one sentence per criterion explaining the score)

Output valid JSON only.
""".strip()

    _DIAGNOSIS_EN = """
You are a diagnostic classifier inside an adaptive reading system.

Given learner state and reading signals, assign exactly one label.

DECISION CRITERIA — apply in order, first match wins:

  underchallenged
    comprehension_score >= 0.85
    AND fluency_score >= 0.75
    AND hint_use_rate <= 0.10
    AND retell_quality >= 0.75

  overloaded
    comprehension_score < 0.50
    OR completion is false
    OR (hint_use_rate >= 0.30 AND retell_quality < 0.50)

  successful_but_support_dependent
    comprehension_score >= 0.70
    AND hint_use_rate >= 0.30

  cohesion_inference_barrier
    comprehension_score >= 0.70
    AND inference_score < 0.55

  vocabulary_barrier
    comprehension_score < 0.70
    AND vocabulary_need score >= syntax_need score

  syntax_barrier
    comprehension_score < 0.70
    AND syntax_need score > vocabulary_need score

  well_calibrated
    all other cases where the learner completed with adequate comprehension

Return: {"diagnosis": "<label>", "reason": "<one sentence>"}
Output valid JSON only.
""".strip()

    # ── Spanish system prompts ───────────────────────────────────────────────

    _CANONICALIZER_ES = """
Eres un extractor de pasajes canónicos dentro de un sistema adaptativo de lectura.

Tu tarea es convertir un pasaje fuente en una representación instruccional protegida.
Debes identificar:
- unidades de significado protegidas,
- restricciones de secuencia protegidas,
- vocabulario obligatorio,
- objetivo instruccional.

Reglas:
- Extrae únicamente los eventos, afirmaciones o ideas fundamentales.
- No incluyas detalles triviales a menos que sean esenciales desde el punto
  de vista instruccional.
- Las unidades de significado deben ser concisas y explícitas.
- Los anclajes (anchors) deben ser palabras que aparezcan TEXTUALMENTE en el
  TEXTO FUENTE — no paráfrasis. El validador los usa para comprobar reescrituras,
  por lo que deben estar presentes de forma literal.
- Las restricciones de secuencia solo deben incluirse cuando el orden importe de
  forma causal o lógica.
- El vocabulario obligatorio debe incluir únicamente los términos esenciales para
  la lección.
- No dividas un único evento en varias unidades de significado a menos que sea
  instruccionalmente necesario evaluarlas de forma independiente. La sobredivisión
  hace que la validación sea frágil y que la puntuación del recuento sea poco fiable.
- Genera únicamente JSON válido.
""".strip()

    _CANDIDATE_GENERATOR_ES = """
Eres un reescritor de pasajes con restricciones dentro de un sistema adaptativo de lectura.

Tu tarea es generar versiones semánticamente equivalentes del mismo pasaje.
Debes preservar:
- todas las unidades de significado obligatorias (todas — omitir alguna es un error),
- el objetivo instruccional,
- el vocabulario obligatorio (usa glosas entre paréntesis cuando gloss_allowed sea true),
- la identidad del pasaje.

Puedes variar:
- la longitud y estructura de las oraciones,
- la carga de vocabulario (sustituye el registro académico por el cotidiano),
- la complejidad sintáctica,
- el soporte de cohesión (frases de transición),
- la segmentación en párrafos (chunking),
- el soporte a la inferencia.

GUÍA DE REESCRITURA SEGÚN EL PERFIL DEL ESTUDIANTE:
Aplica estas reglas según los campos de learner_state:

vocabulary_need HIGH
  Sustituye el vocabulario de registro académico o formal por equivalentes
  del registro cotidiano. Conserva los términos obligatorios; añade una
  definición entre paréntesis cuando gloss_allowed sea true,
  p. ej.: "manuscritos (libros escritos a mano)".

syntax_need HIGH
  Prefiere la voz activa. Divide las cláusulas subordinadas largas en
  oraciones independientes. Evita frases de participio al inicio de la oración.
  Evita las nominalizaciones cuando sea posible (usa "construyeron" en lugar de
  "la construcción de"). Evita la pasiva con se cuando no sea necesaria.

cohesion_need HIGH
  Añade frases de transición explícitas: "Primero, ...", "Por esta razón, ...",
  "Como resultado, ...", "Esto significa que ...", "En consecuencia, ...",
  "Debido a esto, ...". Explicita las relaciones causales en lugar de
  dejarlas implícitas.

support_dependence HIGH / readiness_to_increase LOW
  Aumenta la segmentación (párrafos más cortos). Añade oraciones temáticas
  explícitas. Andamia la inferencia: enuncia las conclusiones que el lector
  podría necesitar inferir.

readiness_to_increase HIGH / support_dependence LOW
  Permite una sintaxis más compleja. Reduce las glosas. Conserva parte del
  vocabulario académico. Deja que el lector realice inferencias razonables
  sin apoyo explícito.

Reglas:
- No elimines ninguna unidad de significado obligatoria. Cada unidad debe
  aparecer en cada versión candidata.
- No cambies la afirmación central ni la estructura de eventos.
- No inviertas las restricciones de secuencia obligatorias.
- Genera exactamente los candidatos indicados en candidate_plan — uno por entrada.
- En llm_self_audit, incluye un objeto "meaning_unit_coverage" que indique el id
  de cada unidad y si está presente en este candidato, p. ej.:
  "meaning_unit_coverage": {"MU1": true, "MU2": true, "MU3": false}
- Genera únicamente JSON válido.
""".strip()

    _FIT_ESTIMATOR_ES = """
Eres un estimador de adecuación texto-estudiante dentro de un sistema adaptativo
de lectura.

Tu tarea es valorar cada candidato en tres dimensiones:
- access (acceso):         probabilidad de que el estudiante comprenda el pasaje
                           de forma adecuada
- growth (desarrollo):     probabilidad de que el pasaje ofrezca un desafío
                           productivo significativo
- support_burden (carga):  medida en que el pasaje depende del andamiaje para
                           tener éxito

Etiquetas: low | medium | high  (sin otros valores)

Reglas:
- Valora cada candidato por separado según learner_state y deterministic_scores.
- Un pasaje muy por encima del nivel del estudiante sin glosas = access: medium o low.
- Un pasaje muy por debajo del nivel del estudiante = growth: low, aunque access sea high.
- Un alto nivel de andamiaje (soporte de vocabulario, cohesión o segmentación)
  aumenta support_burden.
- No des preferencia a los candidatos más sencillos solo porque maximicen el acceso.
  Favorece los candidatos que conserven el desafío productivo cuando el acceso sea
  adecuado.
- Mantén la razón concisa (1-2 oraciones).
- Genera únicamente JSON válido.
""".strip()

    _ASSESSMENT_GENERATOR_ES = """
Eres un generador de evaluaciones dentro de un sistema adaptativo de lectura.

Tu tarea es crear una breve prueba diagnóstica de comprensión para un estudiante
que acaba de leer un pasaje.

Genera a partir del modelo de pasaje canónico; utiliza el pasaje entregado
únicamente para calibrar el vocabulario de las preguntas.
Tu objetivo es producir exactamente las señales que el sistema necesita para
decidir cómo adaptar el siguiente pasaje.

Genera EXACTAMENTE estos seis ítems en este orden:
  Q1  literal_mcq          — evalúa un hecho explícito (apunta a una unidad de
                             significado obligatoria)
  Q2  sequence_mcq         — evalúa el orden temporal o causal de eventos
                             target debe ser {"meaning_unit_ids": ["MUx", "MUy"],
                             "relation": "before"}
  Q3  inference_mcq        — evalúa una causa-efecto o implicación NO enunciada
                             directamente
  Q4  vocabulary_mcq       — evalúa un término de vocabulario obligatorio en contexto
  Q5  retell_short_response — recuento abierto del contenido esencial, puntuado
                              contra las unidades de significado canónicas
  Q6  self_rating          — dificultad percibida por el estudiante (escala 1-5)

Cada ítem MCQ DEBE tener exactamente estos campos:
  id, type, target, question, choices (array de {id, text}), correct_answer

Cada ítem retell DEBE tener:
  id, type, target (null), prompt, rubric ({max_score, criteria: [{points,
  description}]})
  Los criterios del rúbrico deben corresponderse con las unidades de significado
  obligatorias.

Cada ítem self_rating DEBE tener:
  id, type, target (null), prompt, scale ("1-5")

scoring_blueprint DEBE incluir las cuatro claves:
  literal_item_ids, sequence_item_ids, inference_item_ids, vocabulary_item_ids

signal_mapping DEBE incluir:
  comprehension_score: pesos para los cuatro ítems MCQ (suma = 1.0)
  inference_score: pesos para Q3 y Q5 (suma = 1.0)
  vocabulary_signal, retell_quality, difficulty_signal

Reglas:
- Los distractores deben ser plausibles y diagnósticos (cada distractor revela
  un concepto erróneo específico).
- No crees preguntas trampa.
- El rúbrico del recuento debe ser puntuable: cada criterio debe incluir una lista
  "meaning_unit_ids", p. ej. {"points":1,"meaning_unit_ids":["MU1"],"description":"..."}
- El max_score del rúbrico debe ser igual a la suma de los puntos de todos los criterios.
- Los ítems y las opciones de respuesta deben estar redactados en español.
- Genera únicamente JSON válido.
""".strip()

    _RETELL_SCORER_ES = """
Eres un evaluador de recuentos (retell scorer) dentro de un sistema adaptativo
de lectura.

Tu tarea es puntuar el recuento breve de un estudiante según el rúbrico
proporcionado.
Puntúa ÚNICAMENTE según los criterios del rúbrico — no otorgues crédito por
contenido que no esté en el rúbrico.

Evalúa la cobertura de ideas, no la forma lingüística. Acepta:
- Español estándar y español informal o coloquial.
- Respuestas gramaticalmente incompletas que expresen la idea.
- Alternancia de códigos (mezcla de español e inglés) si el contenido de la idea
  es claro.
Penaliza ÚNICAMENTE la ausencia de la idea, no la forma en que está expresada.

Devuelve:
- raw_score         (entero)
- max_score         (entero — repite el max_score del rúbrico)
- matched_meaning_units  (lista de ids de unidades de significado cubiertas)
- matched_relationships  (lista de relaciones causales/temporales identificadas)
- concise_reason    (una oración por criterio explicando la puntuación)

Genera únicamente JSON válido.
""".strip()

    _DIAGNOSIS_ES = """
Eres un clasificador diagnóstico dentro de un sistema adaptativo de lectura.

Dado el estado del estudiante y las señales de lectura, asigna exactamente una
etiqueta.

CRITERIOS DE DECISIÓN — aplica en orden, el primero que coincida gana:

  underchallenged
    comprehension_score >= 0.85
    Y fluency_score >= 0.75
    Y hint_use_rate <= 0.10
    Y retell_quality >= 0.75

  overloaded
    comprehension_score < 0.50
    O completion es false
    O (hint_use_rate >= 0.30 Y retell_quality < 0.50)

  successful_but_support_dependent
    comprehension_score >= 0.70
    Y hint_use_rate >= 0.30

  cohesion_inference_barrier
    comprehension_score >= 0.70
    Y inference_score < 0.55

  vocabulary_barrier
    comprehension_score < 0.70
    Y vocabulary_need score >= syntax_need score

  syntax_barrier
    comprehension_score < 0.70
    Y syntax_need score > vocabulary_need score

  well_calibrated
    todos los demás casos en que el estudiante completó la lectura con una
    comprensión adecuada

IMPORTANTE: Las etiquetas de diagnóstico (diagnosis) deben escribirse en inglés
tal como se indican arriba — son identificadores del sistema, no texto visible
al usuario. El campo reason puede estar en español.

Devuelve: {"diagnosis": "<etiqueta_en_inglés>", "reason": "<una oración>"}
Genera únicamente JSON válido.
""".strip()

    # ── Prompt routing dict ──────────────────────────────────────────────────

    _SYSTEMS: dict[str, dict[str, str]] = {
        "en": {
            "canonicalizer":        _CANONICALIZER_EN,
            "candidate_generator":  _CANDIDATE_GENERATOR_EN,
            "fit_estimator":        _FIT_ESTIMATOR_EN,
            "assessment_generator": _ASSESSMENT_GENERATOR_EN,
            "retell_scorer":        _RETELL_SCORER_EN,
            "diagnosis":            _DIAGNOSIS_EN,
        },
        "es": {
            "canonicalizer":        _CANONICALIZER_ES,
            "candidate_generator":  _CANDIDATE_GENERATOR_ES,
            "fit_estimator":        _FIT_ESTIMATOR_ES,
            "assessment_generator": _ASSESSMENT_GENERATOR_ES,
            "retell_scorer":        _RETELL_SCORER_ES,
            "diagnosis":            _DIAGNOSIS_ES,
        },
    }

    def __init__(self, language: str = "es") -> None:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )
        self.language = language
        p = self._SYSTEMS[language]
        self.CANONICALIZER_SYSTEM       = p["canonicalizer"]
        self.CANDIDATE_GENERATOR_SYSTEM = p["candidate_generator"]
        self.FIT_ESTIMATOR_SYSTEM       = p["fit_estimator"]
        self.ASSESSMENT_GENERATOR_SYSTEM = p["assessment_generator"]
        self.RETELL_SCORER_SYSTEM       = p["retell_scorer"]
        self.DIAGNOSIS_SYSTEM           = p["diagnosis"]

    # ── Backwards-compatible class-level attributes (English defaults) ──────
    # These allow old code that accesses PromptLibrary.SOME_SYSTEM directly
    # (without instantiation) to continue working.
    CANONICALIZER_SYSTEM       = _CANONICALIZER_EN
    CANDIDATE_GENERATOR_SYSTEM = _CANDIDATE_GENERATOR_EN
    FIT_ESTIMATOR_SYSTEM       = _FIT_ESTIMATOR_EN
    ASSESSMENT_GENERATOR_SYSTEM = _ASSESSMENT_GENERATOR_EN
    RETELL_SCORER_SYSTEM       = _RETELL_SCORER_EN
    DIAGNOSIS_SYSTEM           = _DIAGNOSIS_EN

    # ── User prompt builders ─────────────────────────────────────────────────

    @staticmethod
    def canonicalizer_user(source_text: str, passage_id: str, instructional_objective: str) -> str:
        contract = {
            "passage_id": "string",
            "source_text": "string — echo back source_text unchanged",
            "instructional_objective": "string",
            "meaning_units": [
                {
                    "id": "MU1",
                    "text": "concise statement of the core idea",
                    "required": True,
                    "anchors": ["word1_from_source", "word2_from_source"],
                }
            ],
            "sequence_constraints": [{"before": "MU1", "after": "MU2"}],
            "must_preserve_vocabulary": [
                {"term": "exact term from source", "required": True, "gloss_allowed": True}
            ],
        }
        return json.dumps(
            {
                "task": "canonicalize_passage",
                "passage_id": passage_id,
                "instructional_objective": instructional_objective,
                "source_text": source_text,
                "output_contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def candidate_generator_user(
        canonical: "CanonicalPassage",
        learner: "LearnerState",
        candidate_plan: Sequence[dict[str, Any]],
    ) -> str:
        contract = {
            "candidates": [
                {
                    "candidate_id": "A",
                    "passage_id": canonical.passage_id,
                    "relative_band": -1,
                    "text": "the full rewritten passage text",
                    "scaffold": {
                        "vocabulary_support": "low|medium|high",
                        "syntax_support":     "low|medium|high",
                        "cohesion_support":   "low|medium|high",
                        "chunking_support":   "low|medium|high",
                        "inference_support":  "low|medium|high",
                    },
                    "llm_self_audit": {
                        "meaning_preserved":    True,
                        "sequence_preserved":   True,
                        "objective_preserved":  True,
                        "same_passage_identity": True,
                        "notes": "brief self-check notes",
                    },
                }
            ]
        }
        return json.dumps(
            {
                "task": "generate_candidates",
                "canonical_passage": _canonical_to_dict(canonical),
                "learner_state": learner.to_prompt_dict(),
                "candidate_plan": list(candidate_plan),
                "output_contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def fit_estimator_user(
        canonical: "CanonicalPassage",
        learner: "LearnerState",
        validated_candidates: Sequence["CandidatePassage"],
        deterministic_scores: dict[str, dict[str, Any]],
    ) -> str:
        contract = {
            "fit_estimates": [
                {
                    "candidate_id": "A",
                    "access":         "low|medium|high",
                    "growth":         "low|medium|high",
                    "support_burden": "low|medium|high",
                    "reason":         "one or two sentences",
                }
            ]
        }
        return json.dumps(
            {
                "task": "estimate_fit",
                "canonical_passage": _canonical_to_dict(canonical),
                "learner_state": learner.to_prompt_dict(),
                "validated_candidates": [_candidate_to_dict(c) for c in validated_candidates],
                "deterministic_scores": deterministic_scores,
                "output_contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def assessment_generator_user(
        canonical: "CanonicalPassage",
        delivered_passage: "CandidatePassage",
        learner: "LearnerState",
    ) -> str:
        contract = {
            "assessment_blueprint": {
                "passage_id": "string",
                "literal_target":    {"meaning_unit_id": "MU1", "description": "string"},
                "sequence_target":   {"meaning_unit_ids": ["MU1", "MU2"], "description": "string"},
                "inference_target":  {"id": "INF_1", "description": "string"},
                "vocabulary_target": {"term": "string", "description": "string"},
                "gist_target":       {"meaning_unit_ids": ["MU1", "MU2"], "description": "string"},
            },
            "items": [
                {
                    "id": "Q1", "type": "literal_mcq", "target": "MU1",
                    "question": "string",
                    "choices": [{"id": "A", "text": "string"}, {"id": "B", "text": "string"},
                                {"id": "C", "text": "string"}, {"id": "D", "text": "string"}],
                    "correct_answer": "A",
                },
                {
                    "id": "Q2", "type": "sequence_mcq",
                    "target": {"meaning_unit_ids": ["MU1", "MU2"], "relation": "before"},
                    "question": "string",
                    "choices": [{"id": "A", "text": "string"}, {"id": "B", "text": "string"},
                                {"id": "C", "text": "string"}, {"id": "D", "text": "string"}],
                    "correct_answer": "A",
                },
                {
                    "id": "Q3", "type": "inference_mcq", "target": "MU2",
                    "question": "string",
                    "choices": [{"id": "A", "text": "string"}, {"id": "B", "text": "string"},
                                {"id": "C", "text": "string"}, {"id": "D", "text": "string"}],
                    "correct_answer": "A",
                },
                {
                    "id": "Q4", "type": "vocabulary_mcq", "target": "term",
                    "question": "string",
                    "choices": [{"id": "A", "text": "string"}, {"id": "B", "text": "string"},
                                {"id": "C", "text": "string"}, {"id": "D", "text": "string"}],
                    "correct_answer": "A",
                },
                {
                    "id": "Q5", "type": "retell_short_response", "target": None,
                    "prompt": "string",
                    "rubric": {
                        "max_score": 4,
                        "criteria": [
                            {"points": 1, "meaning_unit_ids": ["MU1"], "description": "criterion for MU1"},
                            {"points": 1, "meaning_unit_ids": ["MU2", "MU3"], "description": "criterion for MU2/MU3"},
                            {"points": 1, "meaning_unit_ids": ["MU4"], "description": "criterion for MU4"},
                            {"points": 1, "meaning_unit_ids": ["MU5"], "description": "criterion for MU5"},
                        ],
                    },
                },
                {
                    "id": "Q6", "type": "self_rating", "target": None,
                    "prompt": "string", "scale": "1-5",
                },
            ],
            "scoring_blueprint": {
                "literal_item_ids":    ["Q1"],
                "sequence_item_ids":   ["Q2"],
                "inference_item_ids":  ["Q3"],
                "vocabulary_item_ids": ["Q4"],
            },
            "signal_mapping": {
                "comprehension_score": {
                    "formula": "weighted average",
                    "weights": {"Q1": 0.25, "Q2": 0.25, "Q3": 0.25, "Q4": 0.25},
                },
                "inference_score": {
                    "formula": "weighted average",
                    "weights": {"Q3": 0.6, "Q5": 0.4},
                },
                "vocabulary_signal": {"formula": "Q4 correctness", "weights": {"Q4": 1.0}},
                "retell_quality":    {"formula": "Q5 rubric score normalized 0-1"},
                "difficulty_signal": {"formula": "Q6 raw scale — 1=easy 5=hard"},
            },
        }
        return json.dumps(
            {
                "task": "generate_assessment",
                "canonical_passage": _canonical_to_dict(canonical),
                "delivered_passage": _candidate_to_dict(delivered_passage),
                "learner_state": learner.to_prompt_dict(),
                "output_contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def retell_scorer_user(
        canonical: "CanonicalPassage",
        assessment: "AssessmentPackage",
        learner_response: str,
    ) -> str:
        rubric = next(
            (item.payload.get("rubric") for item in assessment.items
             if item.type == "retell_short_response"),
            {},
        )
        return json.dumps(
            {
                "task": "score_retell",
                "canonical_passage": _canonical_to_dict(canonical),
                "rubric": rubric,
                "learner_response": learner_response,
                "output_contract": {
                    "raw_score": 0,
                    "max_score": 4,
                    "matched_meaning_units": ["MU1"],
                    "matched_relationships": ["CAUSE_1"],
                    "concise_reason": "string",
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def diagnosis_user(learner: "LearnerState", signals: "ReadingSignals") -> str:
        return json.dumps(
            {
                "task": "diagnose_outcome",
                "learner_state": learner.to_prompt_dict(),
                "reading_signals": {
                    "comprehension_score": signals.comprehension_score,
                    "inference_score":     signals.inference_score,
                    "fluency_score":       signals.fluency_score,
                    "hint_use_rate":       signals.hint_use_rate,
                    "reread_count":        signals.reread_count,
                    "difficulty_rating":   signals.difficulty_rating,
                    "retell_quality":      signals.retell_quality,
                    "completion":          signals.completion,
                },
                "output_contract": {
                    "diagnosis": (
                        "underchallenged|well_calibrated|successful_but_support_dependent|"
                        "vocabulary_barrier|syntax_barrier|cohesion_inference_barrier|overloaded"
                    ),
                    "reason": "string",
                },
            },
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Serialization helpers (internal)
# ============================================================

def _canonical_to_dict(canonical: CanonicalPassage) -> dict[str, Any]:
    return {
        "passage_id":             canonical.passage_id,
        "source_text":            canonical.source_text,
        "instructional_objective": canonical.instructional_objective,
        "source_fk":              canonical.source_fk,
        "meaning_units": [
            {"id": mu.id, "text": mu.text, "required": mu.required, "anchors": list(mu.anchors)}
            for mu in canonical.meaning_units
        ],
        "sequence_constraints": [
            {"before": sc.before, "after": sc.after}
            for sc in canonical.sequence_constraints
        ],
        "must_preserve_vocabulary": [
            {"term": vt.term, "required": vt.required, "gloss_allowed": vt.gloss_allowed}
            for vt in canonical.must_preserve_vocabulary
        ],
    }


def _candidate_to_dict(candidate: CandidatePassage) -> dict[str, Any]:
    return {
        "candidate_id":  candidate.candidate_id,
        "passage_id":    candidate.passage_id,
        "relative_band": candidate.relative_band,
        "text":          candidate.text,
        "scaffold":      candidate.scaffold.to_dict(),
        "llm_self_audit": {
            "meaning_preserved":    candidate.llm_self_audit.meaning_preserved,
            "sequence_preserved":   candidate.llm_self_audit.sequence_preserved,
            "objective_preserved":  candidate.llm_self_audit.objective_preserved,
            "same_passage_identity": candidate.llm_self_audit.same_passage_identity,
            "notes":                candidate.llm_self_audit.notes,
        },
    }


# Keep legacy names for callers that import them
canonical_to_prompt_dict = _canonical_to_dict
candidate_to_prompt_dict = _candidate_to_dict


# ============================================================
# Parsing helpers
# ============================================================

def parse_canonical_passage(data: dict[str, Any]) -> CanonicalPassage:
    meaning_units = tuple(
        MeaningUnit(
            id=item["id"],
            text=item["text"],
            required=bool(item.get("required", True)),
            anchors=tuple(item.get("anchors", [])),
        )
        for item in data["meaning_units"]
    )
    sequence_constraints = tuple(
        SequenceConstraint(before=item["before"], after=item["after"])
        for item in data.get("sequence_constraints", [])
    )
    vocab = tuple(
        VocabularyTerm(
            term=item["term"],
            required=bool(item.get("required", True)),
            gloss_allowed=bool(item.get("gloss_allowed", True)),
        )
        for item in data.get("must_preserve_vocabulary", [])
    )
    source_text = data.get("source_text", "")
    return CanonicalPassage(
        passage_id=data["passage_id"],
        source_text=source_text,
        instructional_objective=data["instructional_objective"],
        meaning_units=meaning_units,
        sequence_constraints=sequence_constraints,
        must_preserve_vocabulary=vocab,
        source_fk=data.get("source_fk") or flesch_kincaid_grade(source_text),
    )


def parse_candidate_passages(data: dict[str, Any]) -> list[CandidatePassage]:
    out: list[CandidatePassage] = []
    for item in data["candidates"]:
        s = item["scaffold"]
        a = item["llm_self_audit"]
        out.append(CandidatePassage(
            candidate_id=item["candidate_id"],
            passage_id=item["passage_id"],
            relative_band=int(item["relative_band"]),
            text=item["text"],
            scaffold=ScaffoldProfile(
                vocabulary_support=Level.from_value(s["vocabulary_support"]),
                syntax_support=    Level.from_value(s["syntax_support"]),
                cohesion_support=  Level.from_value(s["cohesion_support"]),
                chunking_support=  Level.from_value(s["chunking_support"]),
                inference_support= Level.from_value(s["inference_support"]),
            ),
            llm_self_audit=SelfAudit(
                meaning_preserved=   bool(a["meaning_preserved"]),
                sequence_preserved=  bool(a["sequence_preserved"]),
                objective_preserved= bool(a["objective_preserved"]),
                same_passage_identity=bool(a["same_passage_identity"]),
                notes=a.get("notes", ""),
                meaning_unit_coverage={
                    str(k): bool(v)
                    for k, v in a.get("meaning_unit_coverage", {}).items()
                } if isinstance(a.get("meaning_unit_coverage"), dict) else {},
            ),
        ))
    return out


def parse_fit_estimates(data: dict[str, Any]) -> dict[str, FitEstimate]:
    out: dict[str, FitEstimate] = {}
    for item in data["fit_estimates"]:
        out[item["candidate_id"]] = FitEstimate(
            access=        Level.from_value(item["access"]),
            growth=        Level.from_value(item["growth"]),
            support_burden=Level.from_value(item["support_burden"]),
            reason=item.get("reason", ""),
        )
    return out


def parse_assessment_package(data: dict[str, Any]) -> AssessmentPackage:
    items: list[AssessmentItem] = []
    for item in data["items"]:
        payload  = dict(item)
        item_id  = payload.pop("id")
        item_type= payload.pop("type")
        target   = payload.pop("target", None)
        items.append(AssessmentItem(id=item_id, type=item_type, target=target, payload=payload))
    return AssessmentPackage(
        assessment_blueprint=data["assessment_blueprint"],
        items=tuple(items),
        scoring_blueprint=data["scoring_blueprint"],
        signal_mapping=data["signal_mapping"],
        raw_json=data,
    )


# ============================================================
# LLM contract validators
# ============================================================

class ValidationError(ALIENError):
    """Raised when an LLM response fails contract validation."""
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(stage=stage, message=message)


def _validate(stage: str, data: dict[str, Any], rules: list[tuple]) -> None:
    """
    Apply a list of (description, bool_expr) validation rules.
    Raises ValidationError with the first failing rule's description.
    """
    for description, check in rules:
        if not check:
            raise ValidationError(stage, description)


def _json_safe(obj: Any) -> Any:
    """
    Recursively coerce Enum values to their .value strings and tuples to lists,
    so the result is safe to pass to json.dumps without a custom encoder.
    Needed because dataclasses.asdict() preserves Enum instances as-is.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def validate_canonical_json(data: dict[str, Any]) -> None:
    mu_ids = {mu.get("id") for mu in data.get("meaning_units", [])}
    _validate("canonicalize_passage", data, [
        ("passage_id must be present",           bool(data.get("passage_id"))),
        ("instructional_objective must be present", bool(data.get("instructional_objective"))),
        ("meaning_units must be non-empty",       bool(data.get("meaning_units"))),
        ("every meaning unit must have id and text",
            all(mu.get("id") and mu.get("text") for mu in data.get("meaning_units", []))),
        ("sequence constraints must reference known MU ids",
            all(sc.get("before") in mu_ids and sc.get("after") in mu_ids
                for sc in data.get("sequence_constraints", []))),
        ("required vocabulary terms must have term field",
            all(vt.get("term") for vt in data.get("must_preserve_vocabulary", []))),
    ])


def validate_candidates_json(data: dict[str, Any], passage_id: str) -> None:
    candidates = data.get("candidates", [])
    _validate("generate_candidates", data, [
        ("candidates list must be non-empty",       bool(candidates)),
        ("all candidates must have candidate_id",   all(c.get("candidate_id") for c in candidates)),
        ("no duplicate candidate_ids",
            len({c.get("candidate_id") for c in candidates}) == len(candidates)),
        ("all candidates must have non-empty text",  all(c.get("text", "").strip() for c in candidates)),
        (f"all candidates must match passage_id {passage_id!r}",
            all(c.get("passage_id") == passage_id for c in candidates)),
        ("all candidates must have scaffold fields",
            all(isinstance(c.get("scaffold"), dict) for c in candidates)),
        ("all candidates must have llm_self_audit fields",
            all(isinstance(c.get("llm_self_audit"), dict) for c in candidates)),
        # meaning_unit_coverage: warn if absent (log) but do not hard-fail — it is a
        # belt-and-suspenders self-audit feature; absence does not corrupt selection.
        # Hard-fail only if present but malformed (not a dict).
        ("meaning_unit_coverage must be a dict when present",
            all(
                c["llm_self_audit"].get("meaning_unit_coverage") is None
                or isinstance(c["llm_self_audit"].get("meaning_unit_coverage"), dict)
                for c in candidates
                if isinstance(c.get("llm_self_audit"), dict)
            )),
    ])


def validate_assessment_json(data: dict[str, Any]) -> None:
    items     = data.get("items", [])
    item_ids  = [i.get("id") for i in items]
    item_types = [i.get("type") for i in items]
    expected_types = [
        "literal_mcq", "sequence_mcq", "inference_mcq",
        "vocabulary_mcq", "retell_short_response", "self_rating",
    ]
    mcq_items = [i for i in items if i.get("type", "").endswith("mcq")]
    retell_items = [i for i in items if i.get("type") == "retell_short_response"]
    sm = data.get("signal_mapping", {})
    weights = sm.get("comprehension_score", {}).get("weights", {})
    weight_sum = sum(weights.values()) if weights else 0.0

    # Retell rubric checks
    retell_rubric_ok = True
    retell_rubric_sum_ok = True
    for r in retell_items:
        rubric = r.get("rubric", {})
        max_sc = rubric.get("max_score")
        criteria = rubric.get("criteria", [])
        if not isinstance(max_sc, int):
            retell_rubric_ok = False
        if criteria:
            criteria_sum = sum(int(c.get("points", 0)) for c in criteria)
            if criteria_sum != max_sc:
                retell_rubric_sum_ok = False

    # Sequence item relational target check
    seq_items = [i for i in items if i.get("type") == "sequence_mcq"]
    seq_target_ok = all(
        isinstance(i.get("target"), dict)
        and isinstance(i["target"].get("meaning_unit_ids"), list)
        and len(i["target"]["meaning_unit_ids"]) >= 2
        for i in seq_items
    ) if seq_items else True

    # Retell criteria must include meaning_unit_ids when present
    retell_mu_ids_ok = all(
        all(isinstance(c.get("meaning_unit_ids"), list) for c in r.get("rubric", {}).get("criteria", []))
        for r in retell_items
    ) if retell_items else True

    _validate("generate_assessment", data, [
        ("exactly 6 assessment items required",     len(items) == 6),
        ("item types must follow fixed blueprint",  item_types == expected_types),
        ("all MCQ items must have correct_answer",
            all("correct_answer" in i for i in mcq_items)),
        ("all MCQ items must have exactly 4 choices",
            all(len(i.get("choices", [])) == 4 for i in mcq_items)),
        ("all MCQ correct_answers must be present in choices",
            all(i.get("correct_answer") in [c.get("id") for c in i.get("choices", [])]
                for i in mcq_items)),
        ("retell rubric must have integer max_score",            retell_rubric_ok),
        ("retell rubric max_score must equal sum of criteria points", retell_rubric_sum_ok),
        ("sequence_mcq target must be relational {meaning_unit_ids, relation}", seq_target_ok),
        ("retell rubric criteria must include meaning_unit_ids lists", retell_mu_ids_ok),
        ("scoring_blueprint must include inference_item_ids",
            "inference_item_ids" in data.get("scoring_blueprint", {})),
        ("comprehension_score weights must be present",  bool(weights)),
        ("comprehension_score weights must sum to ~1.0",
            abs(weight_sum - 1.0) < 0.01 if weights else True),
    ])


def validate_retell_score_json(data: dict[str, Any], max_score: int) -> None:
    """
    Validate a retell-score LLM response.

    The comparison rules are intentionally written with explicit isinstance
    guards rather than relying on _validate's short-circuit, because Python
    evaluates all list elements eagerly when the rules list is constructed.
    A non-integer raw_score would cause a TypeError in the comparison before
    _validate's loop can reach the type check.
    """
    raw = data.get("raw_score")
    mx  = data.get("max_score")
    raw_is_int = isinstance(raw, int) and not isinstance(raw, bool)
    max_is_int = isinstance(mx,  int) and not isinstance(mx,  bool)
    _validate("score_retell", data, [
        ("raw_score must be present",
            "raw_score" in data),
        ("max_score must be present",
            "max_score" in data),
        ("raw_score must be a non-boolean integer",
            raw_is_int),
        ("max_score must be a non-boolean integer",
            max_is_int),
        ("raw_score must be <= max_score",
            (raw_is_int and max_is_int and raw <= mx)),
        ("raw_score must be >= 0",
            (raw_is_int and raw >= 0)),
    ])


def validate_fit_estimates_json(
    data: dict[str, Any], validated_candidate_ids: set[str]
) -> None:
    """
    Validate the fit-estimation LLM response.
    Every validated candidate must have exactly one estimate; labels must be
    low|medium|high; no unknown candidate ids may appear.
    """
    estimates = data.get("fit_estimates", [])
    returned_ids = {e.get("candidate_id") for e in estimates}
    valid_labels  = {"low", "medium", "high"}
    _validate("estimate_fit", data, [
        ("fit_estimates list must be non-empty",
            bool(estimates)),
        ("every validated candidate must have a fit estimate",
            validated_candidate_ids.issubset(returned_ids)),
        ("no duplicate candidate_ids in fit estimates",
            len(returned_ids) == len(estimates)),
        ("no fit estimates for unknown candidates",
            returned_ids.issubset(validated_candidate_ids)),
        ("all access labels must be low|medium|high",
            all(e.get("access") in valid_labels for e in estimates)),
        ("all growth labels must be low|medium|high",
            all(e.get("growth") in valid_labels for e in estimates)),
        ("all support_burden labels must be low|medium|high",
            all(e.get("support_burden") in valid_labels for e in estimates)),
    ])


# ============================================================
# Deterministic semantic checks
# ============================================================

def _unit_anchor_tokens(unit: MeaningUnit) -> set[str]:
    """
    Build the anchor token set for a MeaningUnit.

    Each anchor is split into individual words before normalising, so multi-word
    anchors like 'Antiguo Régimen' or 'derechos individuales' correctly contribute
    both constituent tokens to the set, rather than a single concatenated stem that
    can never match any token produced by content_tokens() on a sentence.

    Without this split, normalize_token('Antiguo Régimen') strips the space and
    produces a fused stem that is never found in sentence token sets, making
    anchor_cov = 0 for all multi-word anchors and degrading MU matching accuracy.
    """
    if unit.anchors:
        out: set[str] = set()
        for anchor in unit.anchors:
            for word_tok in words(anchor):
                norm = normalize_token(word_tok)
                if norm and norm not in _STOPWORDS and len(norm) > 2:
                    out.add(norm)
        if out:
            return out
    return content_tokens(unit.text)


def sentence_unit_match_score(sentence: str, unit: MeaningUnit) -> float:
    sent_toks   = content_tokens(sentence)
    unit_toks   = content_tokens(unit.text)
    anchor_toks = _unit_anchor_tokens(unit)
    if not unit_toks:
        # Unit text has no content tokens (e.g. text is a single stopword or a
        # one-character placeholder).  Return 0.0 — a unit with no tokens should
        # match nothing, not everything.
        #
        # The original code returned 1.0 here, which caused every sentence to
        # score as a perfect match against every MU with empty text.  All MU
        # positions then collapsed to sentence 0, making sequence_ok_from_positions
        # return False for every candidate and forcing DEGRADED (or no) selection.
        return 0.0
    overlap    = ratio(len(unit_toks & sent_toks),   len(unit_toks))
    anchor_cov = ratio(len(anchor_toks & sent_toks), len(anchor_toks))
    neg_pen    = 0.5 if has_negation(sentence) != has_negation(unit.text) else 1.0
    return round(((0.6 * overlap) + (0.4 * anchor_cov)) * neg_pen, 3)


def best_unit_sentence_match(
    sentences: Sequence[str], unit: MeaningUnit
) -> tuple[int | None, float]:
    best_idx, best_score = None, -1.0
    for idx, sent in enumerate(sentences):
        score = sentence_unit_match_score(sent, unit)
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx is None:
        return None, 0.0
    return best_idx, round(best_score, 3)


def meaning_profile(
    candidate_text: str,
    canonical: CanonicalPassage,
    threshold: float,
) -> tuple[float, float, dict[str, int | None]]:
    sents    = split_sentences(candidate_text)
    required = [mu for mu in canonical.meaning_units if mu.required]
    if not required:
        return 1.0, 1.0, {}
    covered, score_sum = 0, 0.0
    positions: dict[str, int | None] = {}
    for mu in required:
        idx, score = best_unit_sentence_match(sents, mu)
        positions[mu.id] = idx if score >= threshold else None
        score_sum += score
        if score >= threshold:
            covered += 1
    coverage  = round(ratio(covered, len(required)), 3)
    avg_score = round(score_sum / len(required), 3)
    return coverage, avg_score, positions


def sequence_ok_from_positions(
    positions: dict[str, int | None], canonical: CanonicalPassage
) -> bool:
    for rule in canonical.sequence_constraints:
        b = positions.get(rule.before)
        a = positions.get(rule.after)
        if b is None or a is None or b >= a:
            return False
    return True


def vocabulary_coverage(candidate_text: str, canonical: CanonicalPassage) -> float:
    """
    Check required vocabulary using stemmed token matching first,
    raw substring as fallback — consistent with meaning_profile.
    """
    required = [vt for vt in canonical.must_preserve_vocabulary if vt.required]
    if not required:
        return 1.0
    candidate_toks = content_tokens(candidate_text)
    text_lower     = candidate_text.lower()
    hits = sum(
        1 for vt in required
        if normalize_token(vt.term.lower()) in candidate_toks
        or vt.term.lower() in text_lower
    )
    return round(ratio(hits, len(required)), 3)


def length_deviation(
    candidate_text: str,
    source_text: str,
    source_fk: float = 0.0,
    target_fk: float = 0.0,
) -> float:
    """
    Deviation of candidate length from *expected* length at the target band.
    When source_fk and target_fk are known, expected length is scaled
    proportionally (a simpler rewrite is expected to be shorter).
    """
    source_count = max(1, len(words(source_text)))
    cand_count   = len(words(candidate_text))
    if source_fk > 0 and target_fk > 0:
        scale    = clamp(target_fk / source_fk, 0.5, 1.5)
        expected = source_count * scale
    else:
        expected = float(source_count)
    return round(abs(cand_count - expected) / max(1.0, expected), 3)


# ============================================================
# Assessment scoring helpers
# ============================================================

def score_mcq(correct_answer: str, learner_answer: str | None) -> float:
    if learner_answer is None:
        return 0.0
    return 1.0 if learner_answer.strip().upper() == correct_answer.strip().upper() else 0.0


def normalize_retell_score(raw_score: int, max_score: int) -> float:
    return 0.0 if max_score <= 0 else round(raw_score / max_score, 3)


def weighted_average(weights: dict[str, float], item_scores: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return round(sum(w * item_scores.get(k, 0.0) for k, w in weights.items()) / total, 3)


# ============================================================
# Deterministic engine
# ============================================================

class DeterministicEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # ── Band arithmetic ──────────────────────────────────────────────────────

    def target_fk(self, learner: LearnerState, relative_band: int) -> float:
        raw = learner.current_band + (relative_band * self.config.band_step)
        return round(clamp(raw, self.config.min_band, self.config.max_band), 2)

    # ── Distance-aware threshold scaling ─────────────────────────────────────

    def _scaled_thresholds(
        self, source_fk: float, learner_band: float
    ) -> tuple[float, float, float]:
        """
        Return (meaning_threshold, vocab_threshold, length_ceiling)
        scaled by passage distance. Closer passages use strict thresholds;
        distant passages relax them so deep rewrites can pass.
        """
        distance = max(0.0, source_fk - learner_band)
        cfg = self.config
        meaning = max(cfg.meaning_floor,
                      cfg.overall_meaning_threshold - distance * cfg.meaning_relax_per_grade)
        vocab   = max(cfg.vocab_floor,
                      cfg.vocabulary_threshold    - distance * cfg.vocab_relax_per_grade)
        length  = min(cfg.length_ceiling,
                      cfg.length_deviation_threshold + distance * cfg.length_relax_per_grade)
        return round(meaning, 3), round(vocab, 3), round(length, 3)

    # ── Candidate plan builder ────────────────────────────────────────────────

    def build_candidate_plan(self, learner: LearnerState) -> list[dict[str, Any]]:
        """
        Derive the candidate plan from learner state.
        Always: safety net (−1), on-level scaffolded (0), push (0 or +1).
        Learner needs determine which scaffold profiles are included.
        """
        plan: list[dict[str, Any]] = [
            {"relative_band": -1, "profile": "light_support"},
        ]
        if learner.vocabulary_need.score >= Level.MEDIUM.score:
            plan.append({"relative_band": 0, "profile": "vocabulary_support"})
        if learner.syntax_need.score >= Level.MEDIUM.score or learner.cohesion_need.score >= Level.MEDIUM.score:
            plan.append({"relative_band": 0, "profile": "syntax_cohesion_support"})
        if not any(p["relative_band"] == 0 for p in plan):
            plan.append({"relative_band": 0, "profile": "light_support"})
        if learner.readiness_to_increase.score >= Level.MEDIUM.score:
            plan.append({"relative_band": 1, "profile": "light_support"})
        else:
            plan.append({"relative_band": 0, "profile": "inference_support"})
        return plan

    # ── Candidate scoring ─────────────────────────────────────────────────────

    def score_candidate(
        self,
        canonical: CanonicalPassage,
        learner: LearnerState,
        candidate: CandidatePassage,
    ) -> DeterministicScores:
        if candidate.passage_id != canonical.passage_id:
            raise ValueError(
                f"Candidate passage_id {candidate.passage_id!r} does not match "
                f"canonical {canonical.passage_id!r}."
            )

        source_fk  = canonical.source_fk or flesch_kincaid_grade(canonical.source_text)
        fk_grade   = flesch_kincaid_grade(candidate.text)
        target_fk  = self.target_fk(learner, candidate.relative_band)
        fk_ok      = abs(fk_grade - target_fk) <= self.config.fk_tolerance

        meaning_thr, vocab_thr, length_ceil = self._scaled_thresholds(
            source_fk, learner.current_band
        )

        meaning_cov, avg_meaning_score, positions = meaning_profile(
            candidate.text, canonical, self.config.unit_sentence_match_threshold
        )
        seq_ok    = sequence_ok_from_positions(positions, canonical)
        vocab_cov = vocabulary_coverage(candidate.text, canonical)
        length_dev = length_deviation(
            candidate.text, canonical.source_text, source_fk, target_fk
        )

        audit = candidate.llm_self_audit
        blocking: list[str] = []
        warnings: list[str] = []

        # Self-audit failures are always blocking
        if not audit.meaning_preserved:     blocking.append("self_audit_meaning")
        if not audit.sequence_preserved:    blocking.append("self_audit_sequence")
        if not audit.objective_preserved:   blocking.append("self_audit_objective")
        if not audit.same_passage_identity: blocking.append("self_audit_identity")

        # Fast-path MU check: if the LLM self-audit provided meaning_unit_coverage,
        # flag any required unit explicitly reported as absent. This runs before the
        # expensive lexical meaning_profile and adds a named blocking reason per MU.
        if audit.meaning_unit_coverage:
            required_mu_ids = {mu.id for mu in canonical.meaning_units if mu.required}
            for mu_id in required_mu_ids:
                if mu_id in audit.meaning_unit_coverage and not audit.meaning_unit_coverage[mu_id]:
                    blocking.append(f"llm_audit_missing_mu({mu_id})")

        # Meaning and sequence are blocking — they protect instructional identity
        if meaning_cov < meaning_thr:
            blocking.append(f"meaning_coverage_low({meaning_cov}<{meaning_thr})")
        if not seq_ok:
            blocking.append("sequence_violated")

        # Surface measures are warnings — they allow degraded selection
        if not fk_ok:
            warnings.append(f"fk_out_of_tolerance(got={fk_grade},target={target_fk})")
        if vocab_cov < vocab_thr:
            warnings.append(f"vocab_coverage_low({vocab_cov}<{vocab_thr})")
        if length_dev > length_ceil:
            warnings.append(f"length_deviation_high({length_dev}>{length_ceil})")

        passed = len(blocking) == 0 and len(warnings) == 0

        return DeterministicScores(
            fk_grade=fk_grade,
            target_fk=target_fk,
            fk_within_tolerance=fk_ok,
            meaning_coverage=meaning_cov,
            avg_meaning_score=avg_meaning_score,
            vocabulary_coverage=vocab_cov,
            length_deviation=length_dev,
            sequence_ok=seq_ok,
            passed_constraints=passed,
            blocking_reasons=tuple(blocking),
            warning_flags=tuple(warnings),
        )

    # ── Candidate selection ───────────────────────────────────────────────────

    def select_candidate(
        self,
        canonical: CanonicalPassage,
        learner: LearnerState,
        candidates: Sequence[CandidatePassage],
        fit_estimates: dict[str, FitEstimate],
        precomputed_scores: dict[str, DeterministicScores] | None = None,
    ) -> tuple[CandidatePassage, DeterministicScores]:
        """
        Select the best candidate using pre-computed scores (no re-scoring).
        Ranking: hardest band → utility → support burden → meaning coverage.
        Falls back to all-passed pool when eligibility filter leaves nothing.
        """
        scores = precomputed_scores or {
            c.candidate_id: self.score_candidate(canonical, learner, c)
            for c in candidates
        }

        rows: list[tuple[CandidatePassage, DeterministicScores, FitEstimate]] = [
            (c, scores[c.candidate_id], fit_estimates[c.candidate_id])
            for c in candidates
            if c.candidate_id in scores and c.candidate_id in fit_estimates
        ]

        passed = [row for row in rows if row[1].passed_constraints]

        if not passed:
            # Degraded selection: no candidate fully passed.
            # Accept candidates with only surface-measure warnings (FK/length/vocab)
            # but no blocking failures (self-audit, meaning coverage, sequence).
            degraded = [row for row in rows if not row[1].blocking_reasons]
            if not degraded:
                raise ALIENError(
                    stage="select_candidate",
                    message=(
                        "No candidates passed validation and none are recoverable. "
                        "All have blocking failures (meaning, sequence, or self-audit). "
                        "Re-generation with stricter instructions is required."
                    ),
                )
            # Among degraded candidates, rank by meaning quality — surface issues are minor
            chosen_row = max(
                degraded,
                key=lambda row: (
                    row[1].meaning_coverage,
                    row[1].avg_meaning_score,
                    row[1].sequence_ok,
                    row[0].relative_band,
                    row[2].utility if row[2] else 0,
                ),
            )
            c, s, _ = chosen_row
            degraded_scores = replace(s, selection_mode=SelectionMode.DEGRADED)
            return c, degraded_scores

        eligible = [
            row for row in passed
            if row[2].access.score        >= self.config.min_access_score
            and row[2].support_burden.score <= self.config.max_support_burden_score
        ]
        pool = eligible or passed

        chosen = max(
            pool,
            key=lambda row: (
                row[0].relative_band,
                row[2].utility,
                -row[2].support_burden.score,
                row[1].meaning_coverage,
                row[1].avg_meaning_score,
                -row[0].scaffold.total_support(),
            ),
        )
        return chosen[0], chosen[1]

    # ── Diagnosis fallback ────────────────────────────────────────────────────

    def diagnose_fallback(self, learner: LearnerState, signals: ReadingSignals) -> DiagnosisLabel:
        cfg = self.config
        if (signals.comprehension_score >= 0.85
                and signals.fluency_score    >= 0.75
                and signals.hint_use_rate    <= cfg.low_hint_use_threshold
                and signals.retell_quality   >= 0.75):
            return DiagnosisLabel.UNDERCHALLENGED

        if (signals.comprehension_score < cfg.severe_comprehension_threshold
                or (signals.hint_use_rate >= cfg.high_hint_use_threshold
                    and signals.retell_quality < 0.50)
                or not signals.completion):
            return DiagnosisLabel.OVERLOADED

        if signals.comprehension_score >= 0.70 and signals.hint_use_rate >= cfg.high_hint_use_threshold:
            return DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT

        if signals.comprehension_score >= 0.70 and signals.inference_score < 0.55:
            return DiagnosisLabel.COHESION_INFERENCE_BARRIER

        if signals.comprehension_score < 0.70:
            return (
                DiagnosisLabel.VOCABULARY_BARRIER
                if learner.vocabulary_need.score >= learner.syntax_need.score
                else DiagnosisLabel.SYNTAX_BARRIER
            )

        return DiagnosisLabel.WELL_CALIBRATED

    # ── Learner-state update ──────────────────────────────────────────────────

    def update_learner_state(
        self,
        learner: LearnerState,
        diagnosis: DiagnosisLabel,
        signals: ReadingSignals,
    ) -> LearnerState:
        updated = learner

        if diagnosis is DiagnosisLabel.UNDERCHALLENGED:
            updated = replace(updated,
                current_band=self.target_fk(updated, 1),
                readiness_to_increase=Level.HIGH,
                support_dependence=updated.support_dependence.down(),
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.WELL_CALIBRATED:
            dep = updated.support_dependence
            if signals.hint_use_rate <= self.config.low_hint_use_threshold:
                dep = dep.down()
            updated = replace(updated,
                readiness_to_increase=updated.readiness_to_increase.up(),
                support_dependence=dep,
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT:
            updated = replace(updated,
                readiness_to_increase=Level.LOW,
                support_dependence=updated.support_dependence.up(),
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.VOCABULARY_BARRIER:
            updated = replace(updated,
                vocabulary_need=updated.vocabulary_need.up(),
                readiness_to_increase=Level.LOW,
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.SYNTAX_BARRIER:
            new_band = updated.current_band
            if signals.comprehension_score < self.config.severe_comprehension_threshold:
                new_band = self.target_fk(updated, -1)
            updated = replace(updated,
                current_band=new_band,
                syntax_need=updated.syntax_need.up(),
                readiness_to_increase=Level.LOW,
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.COHESION_INFERENCE_BARRIER:
            updated = replace(updated,
                cohesion_need=updated.cohesion_need.up(),
                readiness_to_increase=Level.LOW,
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        elif diagnosis is DiagnosisLabel.OVERLOADED:
            updated = replace(updated,
                current_band=self.target_fk(updated, -1),
                vocabulary_need=updated.vocabulary_need.up(),
                syntax_need=updated.syntax_need.up(),
                cohesion_need=updated.cohesion_need.up(),
                support_dependence=updated.support_dependence.up(),
                readiness_to_increase=Level.LOW,
                cycles_on_passage=updated.cycles_on_passage + 1,
            )

        history = (updated.recent_outcomes + (diagnosis,))[-self.config.history_limit:]
        return replace(updated, recent_outcomes=history)


# ============================================================
# Orchestrator
# ============================================================

class AdaptiveReadingSystem:
    def __init__(
        self,
        llm:      LLMBackend,
        engine:   DeterministicEngine | None = None,
        logger:   logging.Logger      | None = None,
        language: str                  = "es",
    ) -> None:
        self.llm     = llm
        self.engine  = engine or DeterministicEngine()
        self.logger  = logger or logging.getLogger(__name__)
        self.prompts = PromptLibrary(language=language)

    # ── Preparation phase ────────────────────────────────────────────────────

    def canonicalize_passage(
        self,
        source_text: str,
        passage_id:  str,
        instructional_objective: str,
    ) -> CanonicalPassage:
        if not source_text or not source_text.strip():
            raise ValueError("source_text must be non-empty.")
        data = self.llm.complete_json(
            self.prompts.CANONICALIZER_SYSTEM,
            PromptLibrary.canonicalizer_user(source_text, passage_id, instructional_objective),
        )
        # Always enforce the original source text — never trust the LLM echo
        data["source_text"] = source_text
        try:
            validate_canonical_json(data)
        except ValidationError as exc:
            raise ALIENError("canonicalize_passage", str(exc), exc) from exc
        return parse_canonical_passage(data)

    def generate_candidates(
        self,
        canonical:      CanonicalPassage,
        learner:        LearnerState,
        candidate_plan: Sequence[dict[str, Any]] | None = None,
    ) -> list[CandidatePassage]:
        plan = candidate_plan if candidate_plan is not None else \
               self.engine.build_candidate_plan(learner)
        data = self.llm.complete_json(
            self.prompts.CANDIDATE_GENERATOR_SYSTEM,
            PromptLibrary.candidate_generator_user(canonical, learner, plan),
        )
        try:
            validate_candidates_json(data, canonical.passage_id)
        except ValidationError as exc:
            raise ALIENError("generate_candidates", str(exc), exc) from exc
        return parse_candidate_passages(data)

    def estimate_fit(
        self,
        canonical:  CanonicalPassage,
        learner:    LearnerState,
        candidates: Sequence[CandidatePassage],
    ) -> tuple[dict[str, FitEstimate], dict[str, DeterministicScores]]:
        """
        Score every candidate exactly once, send validated subset to LLM for fit,
        and return both fit estimates and all scores for downstream use.
        No candidate is scored more than once.
        """
        all_scores: dict[str, DeterministicScores] = {
            c.candidate_id: self.engine.score_candidate(canonical, learner, c)
            for c in candidates
        }
        validated    = [c for c in candidates if all_scores[c.candidate_id].passed_constraints]
        det_for_llm  = {
            cid: _json_safe(dataclasses.asdict(s))
            for cid, s in all_scores.items()
            if all_scores[cid].passed_constraints
        }

        # If nothing passed, send the best-scoring candidates to give LLM
        # something useful to work with (selection will still enforce constraints)
        if not validated:
            self.logger.warning(
                "No candidates passed deterministic validation for passage %s; "
                "sending full pool to fit estimator.",
                canonical.passage_id,
            )
            validated   = list(candidates)
            det_for_llm = {cid: _json_safe(dataclasses.asdict(s)) for cid, s in all_scores.items()}

        try:
            data = self.llm.complete_json(
                self.prompts.FIT_ESTIMATOR_SYSTEM,
                PromptLibrary.fit_estimator_user(
                    canonical, learner, validated, det_for_llm
                ),
            )
            try:
                validate_fit_estimates_json(data, {c.candidate_id for c in validated})
            except ValidationError as vexc:
                raise ALIENError("estimate_fit", str(vexc), vexc) from vexc
            return parse_fit_estimates(data), all_scores
        except ALIENError:
            raise
        except Exception as exc:
            raise ALIENError("estimate_fit", f"LLM fit estimation failed: {exc}", exc) from exc

    def generate_assessment(
        self,
        canonical:        CanonicalPassage,
        delivered_passage: CandidatePassage,
        learner:          LearnerState,
    ) -> AssessmentPackage:
        data = self.llm.complete_json(
            self.prompts.ASSESSMENT_GENERATOR_SYSTEM,
            PromptLibrary.assessment_generator_user(canonical, delivered_passage, learner),
        )
        try:
            validate_assessment_json(data)
        except ValidationError as exc:
            raise ALIENError("generate_assessment", str(exc), exc) from exc
        return parse_assessment_package(data)

    def begin_passage_journey(
        self,
        learner:   LearnerState,
        canonical: CanonicalPassage,
    ) -> LearnerState:
        """
        Initialize journey-tracking fields when a learner starts a new passage.
        - Sets target_band to the source FK of this passage.
        - Sets entry_band to the learner's current band on first assignment.
        - Resets cycles_on_passage to 0 whenever the passage changes.
        Called automatically by prepare_cycle; can also be called explicitly.
        """
        new_passage = learner.target_band != canonical.source_fk
        return replace(
            learner,
            target_band=canonical.source_fk,
            entry_band=learner.current_band if new_passage else learner.entry_band,
            cycles_on_passage=0 if new_passage else learner.cycles_on_passage,
        )

    def prepare_cycle(
        self,
        source_text:             str,
        passage_id:              str,
        instructional_objective: str,
        learner:                 LearnerState,
        candidate_plan:          Sequence[dict[str, Any]] | None = None,
    ) -> CyclePreparation:
        if not source_text or not source_text.strip():
            raise ValueError("source_text must be non-empty.")

        canonical  = self.canonicalize_passage(source_text, passage_id, instructional_objective)
        learner    = self.begin_passage_journey(learner, canonical)
        candidates = self.generate_candidates(canonical, learner, candidate_plan)
        fit_estimates, all_scores = self.estimate_fit(canonical, learner, candidates)

        selected_candidate, selected_scores = self.engine.select_candidate(
            canonical=canonical,
            learner=learner,
            candidates=candidates,
            fit_estimates=fit_estimates,
            precomputed_scores=all_scores,
        )
        assessment = self.generate_assessment(canonical, selected_candidate, learner)

        if selected_scores.selection_mode == SelectionMode.DEGRADED:
            # Distinguish two very different DEGRADED causes so that operators
            # and teachers interpret monitoring dashboards correctly:
            #
            # FK-only DEGRADED: every candidate has surface-measure warnings
            # (legibilidad fuera de tolerancia, desviación de longitud) but no
            # blocking failures.  This is *expected and normal* for passages
            # académicamente densas where required vocabulary is polysyllabic
            # regardless of sentence simplicity.  Selection is still correct.
            # → Operator action: consider raising fk_tolerance in EngineConfig
            #   for this domain (recommended range 2.5–4.0 for academic content).
            #
            # Structural DEGRADED: at least one candidate has a blocking failure
            # (meaning coverage, sequence, self-audit).  The system fell back to
            # the best partially-valid candidate.
            # → Operator action: review LLM generation quality; adjust the
            #   candidate generator system prompt or re-generate.
            fk_only_degraded = (
                all_scores is not None
                and all(
                    not s.blocking_reasons
                    for s in all_scores.values()
                )
            )
            if fk_only_degraded:
                self.logger.warning(
                    "DEGRADED (solo advertencias de superficie) para pasaje %s "
                    "(alumno %s): todos los candidatos tienen solo advertencias de "
                    "medidas superficiales — esto es esperado para vocabulario "
                    "académico en bandas bajas.  Las restricciones de significado y "
                    "secuencia pasaron.  Considere aumentar fk_tolerance en "
                    "EngineConfig para este dominio.  "
                    "Advertencias del candidato seleccionado: %s",
                    canonical.passage_id, learner.learner_id,
                    ", ".join(selected_scores.warning_flags) or "ninguna",
                )
            else:
                self.logger.warning(
                    "DEGRADED (estructural) para pasaje %s (alumno %s): "
                    "ningún candidato pasó todas las restricciones; se usó el "
                    "mejor candidato parcialmente válido.  Revise la calidad de "
                    "generación del LLM.  "
                    "Advertencias del candidato seleccionado: %s",
                    canonical.passage_id, learner.learner_id,
                    ", ".join(selected_scores.warning_flags) or "ninguna",
                )

        return CyclePreparation(
            canonical=canonical,
            selected_candidate=selected_candidate,
            selected_scores=selected_scores,
            assessment=assessment,
            fit_estimates=fit_estimates,
            all_scores=all_scores,
            selection_mode=selected_scores.selection_mode,
            prepared_learner=learner,   # journey fields are set; caller should use this
        )

    # ── Completion phase ─────────────────────────────────────────────────────

    def score_retell(
        self,
        canonical:       CanonicalPassage,
        assessment:      AssessmentPackage,
        learner_response: str,
    ) -> dict[str, Any]:
        retell_item = next(
            (i for i in assessment.items if i.type == "retell_short_response"), None
        )
        rubric = retell_item.payload.get("rubric", {}) if retell_item else {}
        max_score = int(rubric.get("max_score", 4))
        try:
            result = self.llm.complete_json(
                self.prompts.RETELL_SCORER_SYSTEM,
                PromptLibrary.retell_scorer_user(canonical, assessment, learner_response),
            )
            validate_retell_score_json(result, max_score)
            return result
        except Exception as exc:
            self.logger.warning(
                "Retell scoring failed — using deterministic keyword fallback. Error: %s", exc
            )
            return self._retell_fallback(learner_response, rubric, canonical)

    def _retell_fallback(
        self,
        learner_response: str,
        rubric:           dict[str, Any],
        canonical:        CanonicalPassage,
    ) -> dict[str, Any]:
        """
        Deterministic retell scorer: counts rubric criteria whose description
        keywords appear in the learner response. Used when the LLM scorer fails.
        """
        response_toks = content_tokens(learner_response)
        criteria      = rubric.get("criteria", [])
        max_score     = int(rubric.get("max_score", len(criteria) or 4))
        raw_score     = 0
        matched_mus:  list[str] = []

        for crit in criteria:
            desc_toks = content_tokens(crit.get("description", ""))
            overlap   = len(desc_toks & response_toks)
            threshold = max(1, len(desc_toks) // 3)
            if overlap >= threshold:
                raw_score += int(crit.get("points", 1))
                # Extract any MU ids from the description (e.g. "MU1", "MU3")
                for mu in canonical.meaning_units:
                    if mu.id.lower() in crit.get("description", "").lower():
                        matched_mus.append(mu.id)

        raw_score = min(raw_score, max_score)
        return {
            "raw_score":             raw_score,
            "max_score":             max_score,
            "matched_meaning_units": list(set(matched_mus)),
            "matched_relationships": [],
            "concise_reason":        "Evaluación determinista por palabras clave (evaluador LLM no disponible)." if getattr(self, "prompts", None) and getattr(self.prompts, "language", "es") == "es" else "Deterministic keyword fallback (LLM scorer unavailable).",
        }

    def score_assessment(
        self,
        canonical:       CanonicalPassage,
        assessment:      AssessmentPackage,
        learner_answers: dict[str, Any],
    ) -> AssessmentResult:
        item_scores:      dict[str, float] = {}
        difficulty_rating = 0
        retell_quality    = 0.0

        for item in assessment.items:
            if item.type.endswith("mcq"):
                correct = item.payload["correct_answer"]
                answer  = learner_answers.get(item.id)
                item_scores[item.id] = score_mcq(correct, answer)

            elif item.type == "retell_short_response":
                retell_text  = str(learner_answers.get(item.id, "")).strip()
                retell_json  = self.score_retell(canonical, assessment, retell_text)
                raw_score    = int(retell_json["raw_score"])
                max_score    = int(retell_json["max_score"])
                retell_quality = normalize_retell_score(raw_score, max_score)
                item_scores[item.id] = retell_quality

            elif item.type == "self_rating":
                difficulty_rating    = int(learner_answers.get(item.id, 0))
                item_scores[item.id] = float(difficulty_rating)

        sm = assessment.signal_mapping
        comprehension_score = weighted_average(
            sm.get("comprehension_score", {}).get("weights", {}), item_scores
        )

        # Use signal_mapping weights for inference if provided; fall back to hardcode
        infer_weights = sm.get("inference_score", {}).get("weights", {})
        if infer_weights:
            infer_scores = {
                k: (retell_quality if k == "Q5" else item_scores.get(k, 0.0))
                for k in infer_weights
            }
            inference_score = weighted_average(infer_weights, {**item_scores, **infer_scores})
        else:
            inf_ids = assessment.scoring_blueprint.get("inference_item_ids", [])
            inf_avg = sum(item_scores.get(i, 0.0) for i in inf_ids) / max(1, len(inf_ids))
            inference_score = round((inf_avg + retell_quality) / 2, 3)

        vocab_ids    = assessment.scoring_blueprint.get("vocabulary_item_ids", [])
        vocabulary_score = round(
            sum(item_scores.get(i, 0.0) for i in vocab_ids) / max(1, len(vocab_ids)), 3
        )

        return AssessmentResult(
            item_scores=item_scores,
            comprehension_score=comprehension_score,
            inference_score=inference_score,
            vocabulary_score=vocabulary_score,
            retell_quality=retell_quality,
            difficulty_rating=difficulty_rating,
        )

    def build_reading_signals(
        self,
        assessment_result: AssessmentResult,
        fluency_score:     float,
        hint_use_rate:     float,
        reread_count:      int,
        completion:        bool,
    ) -> ReadingSignals:
        return ReadingSignals(
            comprehension_score=assessment_result.comprehension_score,
            inference_score=assessment_result.inference_score,
            fluency_score=round(fluency_score, 3),
            hint_use_rate=round(hint_use_rate, 3),
            reread_count=int(reread_count),
            difficulty_rating=int(assessment_result.difficulty_rating),
            retell_quality=assessment_result.retell_quality,
            completion=bool(completion),
        )

    def diagnose_outcome(self, learner: LearnerState, signals: ReadingSignals) -> DiagnosisLabel:
        try:
            data = self.llm.complete_json(
                self.prompts.DIAGNOSIS_SYSTEM,
                PromptLibrary.diagnosis_user(learner, signals),
            )
            return DiagnosisLabel.from_value(data["diagnosis"])
        except Exception as exc:
            self.logger.warning(
                "LLM diagnosis failed for learner %s — using deterministic fallback. Error: %s",
                learner.learner_id, exc,
            )
            return self.engine.diagnose_fallback(learner, signals)

    def complete_cycle(
        self,
        learner:         LearnerState,
        prep:            CyclePreparation,
        learner_answers: dict[str, Any],
        telemetry:       ReadingTelemetry,
    ) -> CycleOutcome:
        """
        Primary API: accepts CyclePreparation and ReadingTelemetry.
        Uses prep.prepared_learner (journey-initialised) when available so that
        target_band, entry_band, and cycles_on_passage are never lost across the
        prepare → complete boundary.
        """
        # Prefer the journey-initialised snapshot; fall back to external learner arg
        # (which is correct for complete_cycle_flat and other legacy callers).
        effective_learner = prep.prepared_learner if prep.prepared_learner is not None else learner

        assessment_result = self.score_assessment(
            canonical=prep.canonical,
            assessment=prep.assessment,
            learner_answers=learner_answers,
        )
        reading_signals = self.build_reading_signals(
            assessment_result=assessment_result,
            fluency_score=telemetry.fluency_score,
            hint_use_rate=telemetry.hint_use_rate,
            reread_count=telemetry.reread_count,
            completion=telemetry.completion,
        )
        diagnosis = self.diagnose_outcome(effective_learner, reading_signals)
        updated   = self.engine.update_learner_state(effective_learner, diagnosis, reading_signals)
        return CycleOutcome(
            diagnosis=diagnosis,
            updated_learner=updated,
            assessment_result=assessment_result,
            reading_signals=reading_signals,
            cycle_id=prep.cycle_id,
        )

    def complete_cycle_flat(
        self,
        learner:         LearnerState,
        canonical:       CanonicalPassage,
        assessment:      AssessmentPackage,
        learner_answers: dict[str, Any],
        fluency_score:   float,
        hint_use_rate:   float,
        reread_count:    int,
        completion:      bool,
    ) -> CycleOutcome:
        """
        Compatibility wrapper preserving the original flat signature.
        Prefer complete_cycle() for new code.
        """
        return self.complete_cycle(
            learner=learner,
            prep=CyclePreparation(
                canonical=canonical,
                selected_candidate=CandidatePassage(
                    candidate_id="", passage_id=canonical.passage_id,
                    relative_band=0, text="",
                    scaffold=ScaffoldProfile(),
                    llm_self_audit=SelfAudit(True, True, True, True),
                ),
                selected_scores=DeterministicScores(0,0,True,1,1,1,0,True,True),
                assessment=assessment,
                fit_estimates={},
                prepared_learner=learner,  # preserve journey fields on the flat path
            ),
            learner_answers=learner_answers,
            telemetry=ReadingTelemetry(
                fluency_score=fluency_score,
                hint_use_rate=hint_use_rate,
                reread_count=reread_count,
                completion=completion,
            ),
        )


# ============================================================
# Prompt reference (for external tooling / documentation)
# ============================================================

PROMPTS_REFERENCE = {
    "canonicalizer_system":       PromptLibrary._CANONICALIZER_EN,
    "candidate_generator_system": PromptLibrary._CANDIDATE_GENERATOR_EN,
    "fit_estimator_system":       PromptLibrary._FIT_ESTIMATOR_EN,
    "assessment_generator_system": PromptLibrary._ASSESSMENT_GENERATOR_EN,
    "retell_scorer_system":       PromptLibrary._RETELL_SCORER_EN,
    "diagnosis_system":           PromptLibrary._DIAGNOSIS_EN,
    # Spanish variants
    "canonicalizer_system_es":       PromptLibrary._CANONICALIZER_ES,
    "candidate_generator_system_es": PromptLibrary._CANDIDATE_GENERATOR_ES,
    "fit_estimator_system_es":       PromptLibrary._FIT_ESTIMATOR_ES,
    "assessment_generator_system_es": PromptLibrary._ASSESSMENT_GENERATOR_ES,
    "retell_scorer_system_es":       PromptLibrary._RETELL_SCORER_ES,
    "diagnosis_system_es":           PromptLibrary._DIAGNOSIS_ES,
}


# ============================================================
# Spanish deployment configuration
# ============================================================

SPANISH_CONFIG = EngineConfig(
    # Band arithmetic — unchanged
    band_step   = 0.8,
    min_band    = 0.0,
    max_band    = 12.0,
    # FK tolerance — wider because LLM has less precise control over
    # Szigriszt-Pazos score than over FK grade
    fk_tolerance = 1.5,
    # Match threshold — slightly lower because Spanish morphological richness
    # means the stemmer match is noisier than the English stemmer
    unit_sentence_match_threshold = 0.35,
    overall_meaning_threshold     = 0.70,
    vocabulary_threshold          = 0.80,
    # Length — wider because Spanish encodes more information per word
    # (richer morphology → shorter surface forms); rewrites run naturally longer
    length_deviation_threshold    = 0.45,
    # Threshold relaxation per FK grade of passage distance
    meaning_relax_per_grade  = 0.025,
    vocab_relax_per_grade    = 0.03,
    length_relax_per_grade   = 0.025,
    # Hard floors / ceilings
    meaning_floor    = 0.50,
    vocab_floor      = 0.55,
    length_ceiling   = 0.72,
    # Fit estimation eligibility — unchanged
    min_access_score        = 2,
    max_support_burden_score = 2,
    # Diagnosis thresholds — unchanged
    severe_comprehension_threshold = 0.50,
    low_hint_use_threshold         = 0.10,
    high_hint_use_threshold        = 0.30,
    history_limit = 3,
)
"""
Usage:
    from alien_system_es import AdaptiveReadingSystem, SPANISH_CONFIG, DeterministicEngine

    engine = DeterministicEngine(config=SPANISH_CONFIG)
    system = AdaptiveReadingSystem(llm=your_llm, engine=engine, language="es")
"""


if __name__ == "__main__":
    print("ALIEN (edición en español) — AdaptiveReadingSystem module loaded.")
    print("Implement LLMBackend.complete_json(system, user) -> dict to run it.")
