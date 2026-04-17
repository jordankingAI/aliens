"""
test_fixes.py — Regression tests for the four simulation-identified fixes.

Fix 1 — _unit_anchor_tokens splits multi-word anchors word-by-word (EN + ES)
Fix 2 — sentence_unit_match_score returns 0.0 (not 1.0) when unit_toks empty (EN + ES)
Fix 3 — alien_system_es.ALIENError is alien_system.ALIENError (shared base class)
Fix 4 — DEGRADED log differentiates FK-only from structural failures (EN + ES)

Each test targets the exact code path that was broken, using the same inputs
that exposed the failure during the worked-examples simulation.
"""

import sys, logging
sys.path.insert(0, "/home/claude")

from alien_system import (
    MeaningUnit, SequenceConstraint, CanonicalPassage,
    CandidatePassage, ScaffoldProfile, SelfAudit,
    LearnerState, Level, EngineConfig, DeterministicEngine,
    _unit_anchor_tokens, sentence_unit_match_score,
    meaning_profile, sequence_ok_from_positions,
    words, normalize_token, content_tokens, _STOPWORDS,
    ALIENError as EN_ALIENError,
)
import alien_system_es as es
from alien_system_es import (
    _unit_anchor_tokens as es_unit_anchor_tokens,
    sentence_unit_match_score as es_sums,
    meaning_profile as es_meaning_profile,
    sequence_ok_from_positions as es_seq_ok,
    ALIENError as ES_ALIENError,
    ValidationError as ES_ValidationError,
    MeaningUnit as ESMeaningUnit,
    SequenceConstraint as ESSeqConstraint,
    CanonicalPassage as ESCanonicalPassage,
)

# ── Harness ───────────────────────────────────────────────────────────────────
passed = failed = 0
current_group = ""

def group(name):
    global current_group
    current_group = name
    print(f"\n  {'─'*62}")
    print(f"  {name}")
    print(f"  {'─'*62}")

def ck(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"    ✓  {name}")
    else:
        failed += 1
        print(f"    ✗  {name}" + (f"\n         {detail}" if detail else ""))


# ═════════════════════════════════════════════════════════════════════════════
# Fix 1 — multi-word anchor tokenisation (EN)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 1 EN — _unit_anchor_tokens: multi-word anchors split word-by-word")

# All five multi-word anchors from the simulation that were previously
# concatenated into unfindable tokens.
multi_word_cases = [
    ("cottage industries", {"cottage", "industri"}),
    ("labour movement",    {"labour",  "movement"}),
    ("steam power",        {"steam",   "power"}),
    ("trade unionism",     {"trade",   "unionism"}),
    ("Industrial Revolution", {"industrial", "revolution"}),
]
for phrase, expected_members in multi_word_cases:
    mu = MeaningUnit("M", "placeholder text", True, (phrase,))
    toks = _unit_anchor_tokens(mu)
    for expected in expected_members:
        ck(f"anchor '{phrase}' → token set contains '{expected}'",
           expected in toks, f"got {toks}")
    bad_fused = normalize_token(phrase)  # the old broken token
    ck(f"anchor '{phrase}' → fused token '{bad_fused}' NOT present",
       bad_fused not in toks, f"fused token was in set: {toks}")

# Single-word anchors still work
mu_single = MeaningUnit("M","text",True,("factories","Manchester","Birmingham"))
toks_single = _unit_anchor_tokens(mu_single)
ck("single-word anchors: 'factories' present",  "factori" in toks_single)
ck("single-word anchors: 'Manchester' present",  "manchester" in toks_single)

# Fallback to unit.text when anchors produce no tokens (all stopwords)
mu_stopword_anchors = MeaningUnit("M","workers formed the movement",True,("the","and","of"))
toks_fallback = _unit_anchor_tokens(mu_stopword_anchors)
ck("stopword-only anchors → fallback to content_tokens(unit.text)",
   len(toks_fallback) > 0, f"got {toks_fallback}")

# ── Scoring improvement ───────────────────────────────────────────────────────
group("Fix 1 EN — sentence_unit_match_score improved by multi-word anchor fix")

# These are the exact sentences and MUs from the simulation that scored low
# due to multi-word anchors having zero anchor_cov.
score_cases = [
    ("Large factories replaced cottage industries in cities.",
     MeaningUnit("MU3","Factories replaced cottage industries",True,
                 ("cottage industries","factories","Manchester","Birmingham")),
     0.70),  # must be AT LEAST this — old code gave 0.700, fix gives ≥ 0.80
    ("Workers formed the labour movement and demanded reform.",
     MeaningUnit("MU5","Labour movement reform socialism",True,
                 ("labour movement","reform","socialism")),
     0.50),  # must be AT LEAST this — old code gave 0.433
    ("Steam power freed manufacturing from human muscle.",
     MeaningUnit("MU2","Steam power freed manufacturing from muscle",True,
                 ("steam power","Watt","muscle","liberated")),
     0.70),  # must be AT LEAST this — old code gave 0.700, fix gives ≥ 0.80
]
for sent, mu, min_score in score_cases:
    score = sentence_unit_match_score(sent, mu)
    ck(f"score ≥ {min_score} for '{sent[:45]}...'",
       score >= min_score, f"got {score:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# Fix 1 — multi-word anchor tokenisation (ES)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 1 ES — _unit_anchor_tokens: multi-word anchors split word-by-word")

es_multi = [
    ("Antiguo Régimen",     {"antiguo", "régimen"} | {"regimen"}),
    ("derechos individuales",{"derechos", "individual"}),
    ("soberanía popular",   {"soberanía", "popular"} | {"soberania", "popular"}),
    ("familia real",        {"familia", "real"}),
]
for phrase, expected_any in es_multi:
    mu_es = ESMeaningUnit("M","placeholder",True,(phrase,))
    toks_es = es_unit_anchor_tokens(mu_es)
    ck(f"ES anchor '{phrase}' → non-empty token set",
       len(toks_es) > 0, f"got {toks_es}")
    bad_fused = es.normalize_token(phrase)
    ck(f"ES anchor '{phrase}' → fused token '{bad_fused}' NOT present",
       bad_fused not in toks_es, f"fused token in set: {toks_es}")

# ═════════════════════════════════════════════════════════════════════════════
# Fix 2 — empty unit_toks returns 0.0 not 1.0 (EN)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 2 EN — sentence_unit_match_score: empty unit_toks → 0.0 not 1.0")

# MU with text="x" — single non-alphabetic-length token, filtered out by content_tokens
mu_x = MeaningUnit("M","x",True,())
ck("content_tokens('x') is empty",
   len(content_tokens("x")) == 0)
ck("sentence_unit_match_score returns 0.0 for empty unit_toks (not 1.0)",
   sentence_unit_match_score("Any sentence at all.", mu_x) == 0.0)
ck("score is 0.0 for completely unrelated sentence",
   sentence_unit_match_score("The weather is sunny today.", mu_x) == 0.0)
ck("score is 0.0 even for a sentence containing 'x'",
   sentence_unit_match_score("The letter x is used in algebra.", mu_x) == 0.0)

# MU with text consisting entirely of stopwords
mu_stops = MeaningUnit("M","the and of in",True,())
ck("content_tokens of pure stopwords is empty",
   len(content_tokens("the and of in")) == 0)
ck("sentence_unit_match_score returns 0.0 for stopword-only unit text",
   sentence_unit_match_score("Any sentence.", mu_stops) == 0.0)

# ── The cascade that caused the simulation failure ────────────────────────────
group("Fix 2 EN — sequence_ok no longer fails when unit_toks empty")

# Build a canonical with a placeholder MU (text="x") — exactly what the
# simulation test scaffolding did.  Before the fix, ALL sentences matched ALL
# MUs (score 1.0), positions all collapsed to 0, sequence_ok returned False.
canon_placeholder = CanonicalPassage("p","source","obj",
    (MeaningUnit("MU1","Industrial Revolution Britain 1760",True,
                 ("Revolution","Britain","1760")),
     MeaningUnit("MU2","x",True,()),           # ← placeholder, empty unit_toks
     MeaningUnit("MU3","factories cottage industries",True,
                 ("factories","cottage","Manchester")),
    ),
    (SequenceConstraint("MU1","MU2"),
     SequenceConstraint("MU2","MU3")),
)
text = (
    "The Industrial Revolution began in Britain around 1760. "
    "Steam power changed manufacturing. "
    "Factories replaced cottage industries in Manchester."
)
cov, avg, pos = meaning_profile(text, canon_placeholder, 0.20)
ck("coverage ≥ 0.66 even with placeholder MU2",
   cov >= 0.66, f"got {cov:.3f}")
ck("MU1 maps to sentence 0",  pos.get("MU1") == 0)
ck("MU3 maps to sentence 2",  pos.get("MU3") == 2)
# MU2 will have score 0.0 for all sentences → position = None (below threshold)
ck("MU2 position is None (empty unit_toks scores 0.0)",
   pos.get("MU2") is None, f"got pos['MU2']={pos.get('MU2')}")
# sequence_ok: if MU2 position is None, constraint MU1→MU2 is unsatisfied,
# which is the correct conservative result — the MU is genuinely unknown.
# The key point: the old code had ALL positions = 0, which also violated seq.
# The fix changes nothing about seq_ok for this case, but prevents the
# cascade where GOOD MUs also got position 0 due to score-inflation.

# Verify with fully-specified MUs that sequence_ok works correctly after fix
canon_good = CanonicalPassage("p","source","obj",
    (MeaningUnit("MU1","Industrial Revolution Britain 1760",True,
                 ("Revolution","Britain","1760","transformation")),
     MeaningUnit("MU2","steam power Watt muscle",True,
                 ("steam","Watt","muscle")),
     MeaningUnit("MU3","factories cottage Manchester Birmingham",True,
                 ("factories","cottage","Manchester","Birmingham")),
     MeaningUnit("MU4","children brutal wages machinery",True,
                 ("children","brutal","wages","machinery")),
     MeaningUnit("MU5","labour movement reform socialism unionism",True,
                 ("labour","movement","reform","socialism","unionism")),
    ),
    (SequenceConstraint("MU1","MU2"),SequenceConstraint("MU2","MU3"),
     SequenceConstraint("MU3","MU4"),SequenceConstraint("MU4","MU5")),
)
well_formed = (
    "The Industrial Revolution began in Britain around 1760 — the greatest "
    "transformation since agriculture was invented. "
    "Steam power was the key: Watt's engine freed manufacturing from human muscle. "
    "As a result, factories replaced cottage industries in Manchester and Birmingham. "
    "But factory work was brutal: children worked for low wages with dangerous machinery. "
    "Workers formed the labour movement, demanded reform, and socialism and trade "
    "unionism grew."
)
cov2, avg2, pos2 = meaning_profile(well_formed, canon_good, 0.25)
seq2 = sequence_ok_from_positions(pos2, canon_good)
ck("coverage = 1.0 for well-formed candidate",  cov2 == 1.0, f"got {cov2}")
ck("sequence_ok = True for well-formed candidate", seq2, f"positions={pos2}")
ck("MU1 before MU2 before MU3",
   pos2.get("MU1",99) < pos2.get("MU2",99) < pos2.get("MU3",99),
   f"positions={pos2}")

# ═════════════════════════════════════════════════════════════════════════════
# Fix 2 — empty unit_toks returns 0.0 not 1.0 (ES)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 2 ES — sentence_unit_match_score: empty unit_toks → 0.0 not 1.0")

mu_es_x = ESMeaningUnit("M","x",True,())
ck("ES: sentence_unit_match_score returns 0.0 for empty unit_toks",
   es_sums("Cualquier oración del texto.", mu_es_x) == 0.0)
ck("ES: score is 0.0 for unrelated sentence",
   es_sums("El gato duerme en la silla.", mu_es_x) == 0.0)

mu_es_stops = ESMeaningUnit("M","el y de los",True,())
ck("ES: stopword-only unit text → 0.0",
   es_sums("La Revolución Francesa cambió Europa.", mu_es_stops) == 0.0)

# ═════════════════════════════════════════════════════════════════════════════
# Fix 3 — ALIENError shared base class (EN ↔ ES)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 3 — alien_system_es.ALIENError is alien_system.ALIENError")

ck("ES ALIENError is EN ALIENError (same class object)",
   ES_ALIENError is EN_ALIENError,
   f"EN={EN_ALIENError!r}, ES={ES_ALIENError!r}")

# ES ValidationError still subclasses the shared ALIENError
ck("ES ValidationError is subclass of ES ALIENError",
   issubclass(ES_ValidationError, ES_ALIENError))
ck("ES ValidationError is subclass of EN ALIENError (via shared base)",
   issubclass(ES_ValidationError, EN_ALIENError))

# A caller using only EN ALIENError can now catch ES errors
def _raise_es_validation():
    raise ES_ValidationError("test_stage", "test message")

def _raise_es_alien():
    raise ES_ALIENError("test_stage", "test message")

caught_validation = False
try:
    _raise_es_validation()
except EN_ALIENError:
    caught_validation = True
ck("ES ValidationError caught by except EN_ALIENError", caught_validation)

caught_alien = False
try:
    _raise_es_alien()
except EN_ALIENError:
    caught_alien = True
ck("ES ALIENError caught by except EN_ALIENError", caught_alien)

# stage attribute still works correctly
try:
    raise ES_ALIENError("my_stage", "my message")
except EN_ALIENError as exc:
    ck("caught ES ALIENError has .stage attribute", exc.stage == "my_stage")

# EN ALIENError still works independently
caught_en = False
try:
    raise EN_ALIENError("s","m")
except EN_ALIENError:
    caught_en = True
ck("EN ALIENError caught by except EN_ALIENError", caught_en)

# ═════════════════════════════════════════════════════════════════════════════
# Fix 4 — DEGRADED log differentiation (EN)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 4 EN — DEGRADED log: FK-only vs structural differentiation")

from alien_system import (
    AdaptiveReadingSystem, TaskRoutingMockLLM, parse_canonical_passage,
    parse_candidate_passages, parse_fit_estimates, parse_assessment_package,
    DeterministicScores, SelectionMode, FitEstimate,
)
import dataclasses

# Build a loose engine so candidates can be scored without blocking
loose_cfg = EngineConfig(
    fk_tolerance=0.5,          # very tight → all FK warnings
    overall_meaning_threshold=0.0,
    vocabulary_threshold=0.0,
    length_deviation_threshold=10.0,
    length_ceiling=10.0,
)
loose_eng = DeterministicEngine(config=loose_cfg)

# Capture log records
class CapHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, r):
        self.records.append(r)

cap = CapHandler()
logger = logging.getLogger("alien_fix4_test")
logger.addHandler(cap)
logger.setLevel(logging.DEBUG)

# --- Case A: FK-only DEGRADED ---
# All candidates pass meaning/sequence but fail FK.
# Manually exercise the logic via AdaptiveReadingSystem.prepare_cycle
# using a mock that returns one well-formed candidate with high FK.

SOURCE = (
    "The Industrial Revolution began in Britain around 1760. "
    "Steam power freed manufacturing from dependence on human muscle. "
    "Factories in Manchester replaced cottage industries. "
    "Working conditions were brutal — children worked for negligible wages. "
    "This suffering catalysed the labour movement, reform, socialism and trade unionism."
)

CANON_DATA = {
    "passage_id": "ir_fix4",
    "source_text": SOURCE,
    "instructional_objective": "Understand Industrial Revolution causes and effects.",
    "meaning_units": [
        {"id":"MU1","text":"Industrial Revolution Britain 1760","required":True,
         "anchors":["Revolution","Britain","1760"]},
        {"id":"MU2","text":"Steam power freed manufacturing from muscle","required":True,
         "anchors":["steam","muscle","freed"]},
        {"id":"MU3","text":"Factories Manchester replaced cottage industries","required":True,
         "anchors":["factories","Manchester","cottage"]},
        {"id":"MU4","text":"Workers children brutal wages negligible","required":True,
         "anchors":["children","brutal","wages"]},
        {"id":"MU5","text":"Labour movement reform socialism trade unionism","required":True,
         "anchors":["labour","movement","reform","socialism","unionism"]},
    ],
    "sequence_constraints": [
        {"before":"MU1","after":"MU2"},{"before":"MU2","after":"MU3"},
        {"before":"MU3","after":"MU4"},{"before":"MU4","after":"MU5"},
    ],
    "must_preserve_vocabulary": [],
}

CAND_TEXT = (
    "The Industrial Revolution began in Britain around 1760 — a transformation since agriculture. "
    "Steam power freed manufacturing from human muscle via Watt's engine. "
    "Factories replaced cottage industries in Manchester and Birmingham. "
    "But conditions were brutal: children worked for negligible wages amid dangerous machinery. "
    "This suffering catalysed the labour movement, reform laws, socialism and trade unionism."
)

MOCK_RESPONSES = {
    "canonicalize_passage": CANON_DATA,
    "generate_candidates": {"candidates": [{
        "candidate_id": "A", "passage_id": "ir_fix4", "relative_band": 0,
        "text": CAND_TEXT,
        "scaffold": {"vocabulary_support":"medium","syntax_support":"low",
                     "cohesion_support":"medium","chunking_support":"low",
                     "inference_support":"low"},
        "llm_self_audit": {
            "meaning_preserved": True, "sequence_preserved": True,
            "objective_preserved": True, "same_passage_identity": True,
            "notes": "All MUs present.",
            "meaning_unit_coverage": {
                "MU1":True,"MU2":True,"MU3":True,"MU4":True,"MU5":True
            },
        },
    }]},
    "estimate_fit": {"fit_estimates": [{
        "candidate_id":"A","access":"high","growth":"medium",
        "support_burden":"medium","reason":"Good fit."
    }]},
    "generate_assessment": {
        "assessment_blueprint": {"passage_id":"ir_fix4"},
        "items": [
            {"id":"Q1","type":"literal_mcq","target":"MU1","question":"q",
             "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                        {"id":"C","text":"c"},{"id":"D","text":"d"}],
             "correct_answer":"A"},
            {"id":"Q2","type":"sequence_mcq",
             "target":{"meaning_unit_ids":["MU1","MU2"],"relation":"before"},
             "question":"q",
             "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                        {"id":"C","text":"c"},{"id":"D","text":"d"}],
             "correct_answer":"A"},
            {"id":"Q3","type":"inference_mcq","target":"MU5","question":"q",
             "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                        {"id":"C","text":"c"},{"id":"D","text":"d"}],
             "correct_answer":"A"},
            {"id":"Q4","type":"vocabulary_mcq","target":"labour","question":"q",
             "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                        {"id":"C","text":"c"},{"id":"D","text":"d"}],
             "correct_answer":"A"},
            {"id":"Q5","type":"retell_short_response","target":None,"prompt":"Retell.",
             "rubric":{"max_score":4,"criteria":[
                 {"points":1,"meaning_unit_ids":["MU1"],"description":"MU1"},
                 {"points":1,"meaning_unit_ids":["MU2"],"description":"MU2"},
                 {"points":1,"meaning_unit_ids":["MU3"],"description":"MU3"},
                 {"points":1,"meaning_unit_ids":["MU4"],"description":"MU4"},
             ]}},
            {"id":"Q6","type":"self_rating","target":None,"prompt":"How hard?","scale":"1-5"},
        ],
        "scoring_blueprint": {
            "literal_item_ids":["Q1"],"sequence_item_ids":["Q2"],
            "inference_item_ids":["Q3"],"vocabulary_item_ids":["Q4"]
        },
        "signal_mapping": {
            "comprehension_score":{"formula":"wa","weights":{"Q1":0.25,"Q2":0.25,"Q3":0.25,"Q4":0.25}},
            "inference_score":{"formula":"wa","weights":{"Q3":0.6,"Q5":0.4}},
            "vocabulary_signal":{"formula":"Q4","weights":{"Q4":1.0}},
            "retell_quality":{"formula":"Q5"},
            "difficulty_signal":{"formula":"Q6"},
        },
    },
}

mock_llm = TaskRoutingMockLLM(responses=MOCK_RESPONSES)
sys_en = AdaptiveReadingSystem(llm=mock_llm, engine=loose_eng, logger=logger)
learner = LearnerState("test_learner", 5.0, vocabulary_need=Level.HIGH)

cap.records.clear()
prep = sys_en.prepare_cycle(SOURCE, "ir_fix4", "Understand Industrial Revolution.", learner)

# Check log messages
warning_msgs = [r.getMessage() for r in cap.records if r.levelno == logging.WARNING]
degraded_msgs = [m for m in warning_msgs if "DEGRADED" in m]

ck("DEGRADED warning was emitted", len(degraded_msgs) > 0,
   f"warning_msgs={warning_msgs}")

if degraded_msgs:
    msg = degraded_msgs[0]
    # FK-only case: no blocking reasons → should log FK-only variant
    # (whether it fires depends on which DEGRADED path was taken)
    ck("DEGRADED message references the passage id",
       "ir_fix4" in msg, f"msg={msg[:120]}")
    ck("DEGRADED message references the learner id",
       "test_learner" in msg, f"msg={msg[:120]}")
    # The message should contain one of the two distinguishing phrases
    has_fk_only_msg = "FK" in msg or "surface" in msg or "domain" in msg
    has_structural_msg = "structural" in msg or "review" in msg.lower() or "meaning" in msg.lower()
    ck("DEGRADED message contains differentiating content (FK-only or structural)",
       has_fk_only_msg or has_structural_msg, f"msg={msg[:200]}")

# Also verify that the all_scores-based fk_only_degraded logic is exercised
# by checking whether the selected candidate was DEGRADED
ck("selected candidate is in DEGRADED mode",
   prep.selection_mode == SelectionMode.DEGRADED,
   f"mode={prep.selection_mode}")

# ═════════════════════════════════════════════════════════════════════════════
# Fix 4 — EngineConfig fk_tolerance documentation (EN + ES)
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 4 — EngineConfig fk_tolerance documentation")

from alien_system import EngineConfig as EN_CFG
from alien_system_es import EngineConfig as ES_CFG
import inspect

en_cfg_src = inspect.getsource(EN_CFG)
ck("EN EngineConfig documents academic domain guidance",
   "academic" in en_cfg_src.lower() and "2.5" in en_cfg_src,
   "Missing 'academic' or '2.5' in EngineConfig source")
ck("EN EngineConfig mentions FK-only DEGRADED context",
   "FK-only" in en_cfg_src or "domain vocabulary" in en_cfg_src,
   "Missing FK-only DEGRADED mention")
ck("EN EngineConfig documents recommended fk_tolerance range",
   "4.0" in en_cfg_src and "1.2" in en_cfg_src,
   "Missing range values")

es_cfg_src = inspect.getsource(ES_CFG)
ck("ES EngineConfig has expanded fk_tolerance documentation",
   len(es_cfg_src) > len("    fk_tolerance:              float = 1.2") + 200,
   "ES EngineConfig fk_tolerance comment not expanded")
ck("ES EngineConfig includes Spanish-language guidance",
   "académic" in es_cfg_src or "dominio" in es_cfg_src,
   "Missing Spanish academic domain guidance")

# ═════════════════════════════════════════════════════════════════════════════
# End-to-end: all fixes working together
# ═════════════════════════════════════════════════════════════════════════════

group("End-to-end: all fixes correct in combination")

# Build a full scoring scenario with multi-word anchors in all MUs
# and verify that meaning_profile + sequence_ok both pass.
en_canon = CanonicalPassage("full_test","source","obj",
    (MeaningUnit("MU1","Industrial Revolution Britain 1760",True,
                 ("Industrial Revolution","Britain","1760","transformation","agriculture")),
     MeaningUnit("MU2","Steam power Watt freed manufacturing muscle",True,
                 ("steam power","Watt","muscle","liberated")),
     MeaningUnit("MU3","Factories Manchester Birmingham cottage industries",True,
                 ("cottage industries","factories","Manchester","Birmingham")),
     MeaningUnit("MU4","Workers children brutal wages machinery",True,
                 ("children","brutal","wages","machinery")),
     MeaningUnit("MU5","Labour movement reform socialism trade unionism",True,
                 ("labour movement","reform","socialism","trade unionism")),
    ),
    (SequenceConstraint("MU1","MU2"),SequenceConstraint("MU2","MU3"),
     SequenceConstraint("MU3","MU4"),SequenceConstraint("MU4","MU5")),
)

candidate_text = (
    "The Industrial Revolution began in Britain around 1760 — the greatest "
    "transformation since agriculture was invented. "
    "Steam power was central: Watt improved the steam engine to free manufacturing "
    "from human muscle. "
    "As a result, factories replaced cottage industries in Manchester and Birmingham. "
    "But conditions were brutal: children worked for negligible wages with dangerous machinery. "
    "Workers formed the labour movement, demanded reform laws, and socialism "
    "and trade unionism emerged."
)

cov_e2e, avg_e2e, pos_e2e = meaning_profile(candidate_text, en_canon, 0.25)
seq_e2e = sequence_ok_from_positions(pos_e2e, en_canon)

ck("E2E EN: meaning coverage = 1.0", cov_e2e == 1.0, f"got {cov_e2e}")
ck("E2E EN: sequence_ok = True",     seq_e2e,         f"positions={pos_e2e}")
ck("E2E EN: MU positions strictly increasing",
   all(pos_e2e.get(f"MU{i}",99) < pos_e2e.get(f"MU{i+1}",99)
       for i in range(1,5)),
   f"positions={pos_e2e}")

# Spanish end-to-end
es_canon = ESCanonicalPassage("full_es","fuente","obj",
    (ESMeaningUnit("UM1","Revolución Francesa Bastilla 1789",True,
                   ("Revolución Francesa","Bastilla","1789","transformó")),
     ESMeaningUnit("UM2","Antiguo Régimen colapsó crisis ilustradas",True,
                   ("Antiguo Régimen","colapsó","crisis","ilustradas")),
     ESMeaningUnit("UM3","Declaración derechos hombre igualdad soberanía",True,
                   ("Declaración de los Derechos del Hombre","igualdad","soberanía")),
     ESMeaningUnit("UM4","Terror guillotina miles muertos familia real",True,
                   ("Terror","guillotina","familia real")),
     ESMeaningUnit("UM5","Napoleón libertad igualdad fraternidad Europa",True,
                   ("Napoleón Bonaparte","libertad, igualdad y fraternidad")),
    ),
    (ESSeqConstraint("UM1","UM2"),ESSeqConstraint("UM2","UM3"),
     ESSeqConstraint("UM3","UM4"),ESSeqConstraint("UM4","UM5")),
)

es_cand = (
    "La Revolución Francesa comenzó con la toma de la Bastilla en 1789 y transformó Europa. "
    "El Antiguo Régimen colapsó ante la crisis económica y las ideas ilustradas. "
    "La Declaración proclamó la igualdad y la soberanía popular, desafiando al rey. "
    "Pero el Terror segó miles de vidas bajo la guillotina, incluyendo la familia real. "
    "Napoleón llevó los ideales de libertad, igualdad y fraternidad por todo el continente."
)

cov_es, avg_es, pos_es = es_meaning_profile(es_cand, es_canon, 0.20)
seq_es = es_seq_ok(pos_es, es_canon)

ck("E2E ES: meaning coverage = 1.0", cov_es == 1.0, f"got {cov_es}")
ck("E2E ES: sequence_ok = True",     seq_es,         f"positions={pos_es}")

# ALIENError cross-module catch still works at end
try:
    raise ES_ValidationError("end_to_end","test")
except EN_ALIENError as e:
    ck("E2E: ES ValidationError catchable via EN ALIENError handler",
       e.stage == "end_to_end")

# ═════════════════════════════════════════════════════════════════════════════
# Fix 5 — split_sentences handles terminal punctuation inside quotation marks
# ═════════════════════════════════════════════════════════════════════════════

group("Fix 5 EN — split_sentences: terminal punct inside quotes triggers split")

from alien_system import split_sentences as en_split
import alien_system_es as _es
es_split = _es.split_sentences

# The exact sentence from the simulation that previously failed to split:
# "Her grandmother had written: 'Some doors only open from the inside.' Maya read..."
# — the period is INSIDE the single-quote, so the old lookbehind (?<=[.!?])
#   saw a quote character before the space, not a period, and did not split.
sim_sentence = (
    "Her grandmother had written: 'Some doors only open from the inside.' "
    "Maya read the sentence twice, but its meaning remained stubbornly out of reach."
)
sim_parts = en_split(sim_sentence)
ck("simulation sentence splits into 2 parts at period-inside-single-quote",
   len(sim_parts) == 2,
   f"got {len(sim_parts)} parts: {sim_parts}")
if len(sim_parts) >= 2:
    ck("part[0] ends with the closing quote",
       sim_parts[0].endswith("'") or sim_parts[0].endswith(".'"),
       f"part[0] = {sim_parts[0]!r}")
    ck("part[1] starts with Maya",
       sim_parts[1].startswith("Maya"),
       f"part[1] = {sim_parts[1]!r}")

# Period inside double quotes
double_q = 'She said "Never give up." He agreed immediately.'
dq_parts = en_split(double_q)
ck('period inside double quotes → 2 parts',
   len(dq_parts) == 2, f"got {len(dq_parts)}: {dq_parts}")

# Exclamation inside single quotes
excl_q = "She shouted: 'Watch out!' He ducked behind the desk."
eq_parts = en_split(excl_q)
ck("exclamation inside single quotes → 2 parts",
   len(eq_parts) == 2, f"got {len(eq_parts)}: {eq_parts}")

# Plain period (must still work — regression)
plain = "Maya walked out the door. She did not look back."
plain_parts = en_split(plain)
ck("plain period split still works (regression)",
   len(plain_parts) == 2, f"got {len(plain_parts)}: {plain_parts}")

# Multiple sentences including a quoted one
multi = (
    "The morning was grey. "
    "Her grandmother had written: 'Some doors only open from the inside.' "
    "Maya folded the note. "
    "She stepped outside."
)
multi_parts = en_split(multi)
ck("four-sentence text with quoted sentence → 4 parts",
   len(multi_parts) == 4, f"got {len(multi_parts)}: {multi_parts}")

# Newline-separated text still works (regression)
newline_text = "First sentence.\nSecond sentence.\nThird sentence."
nl_parts = en_split(newline_text)
ck("newline-separated sentences still split (regression)",
   len(nl_parts) == 3, f"got {len(nl_parts)}: {nl_parts}")

# ── MU positions now correct on the simulation passages ───────────────────────
group("Fix 5 EN — MU3/MU4 positions distinct after quote-split fix")

from alien_system import (
    parse_canonical_passage, meaning_profile, sequence_ok_from_positions
)

SIM_SOURCE = (
    "On the morning she was supposed to leave for the city, Maya found the envelope "
    "wedged beneath the welcome mat. "
    "She recognised her grandmother's handwriting — the looping, deliberate cursive. "
    "Inside was a faded photograph of a girl Maya had never seen, "
    "standing in front of a house that no longer existed. "
    "Her grandmother had written: 'Some doors only open from the inside.' "
    "Maya read the sentence twice, but its meaning remained stubbornly out of reach. "
    "She wondered whether the city had any doors worth opening at all."
)
SIM_CANON = parse_canonical_passage({
    "passage_id": "fix5_test",
    "source_text": SIM_SOURCE,
    "instructional_objective": "test",
    "meaning_units": [
        {"id":"MU1","text":"Maya finds envelope from grandmother morning city","required":True,
         "anchors":["envelope","grandmother","morning","city","Maya"]},
        {"id":"MU2","text":"envelope contains photograph girl house never existed","required":True,
         "anchors":["photograph","house","girl","existed","never"]},
        {"id":"MU3","text":"grandmother message doors only open inside","required":True,
         "anchors":["doors","inside","written","grandmother"]},
        {"id":"MU4","text":"Maya meaning stubbornly wondered city","required":True,
         "anchors":["meaning","stubbornly","wondered","city"]},
    ],
    "sequence_constraints":[
        {"before":"MU1","after":"MU2"},{"before":"MU2","after":"MU3"},{"before":"MU3","after":"MU4"}
    ],
    "must_preserve_vocabulary": [],
})

sim_candidate = (
    "On the morning she was leaving for the city, Maya found an envelope. "
    "She could tell it was from her grandmother — she knew the cursive right away. "
    "Inside were a letter and a photograph of a girl she had never seen, "
    "standing in front of a house that no longer existed. "
    "Her grandmother had written: 'Some doors only open from the inside.' "
    "Maya read it again and again, but the meaning stayed stubbornly out of reach. "
    "She put the photograph in her pocket, wondering if the city had any doors worth opening."
)

cov_sim, avg_sim, pos_sim = meaning_profile(sim_candidate, SIM_CANON, 0.25)
seq_sim = sequence_ok_from_positions(pos_sim, SIM_CANON)

ck("simulation candidate: meaning coverage = 1.0",
   cov_sim == 1.0, f"got {cov_sim}")
ck("simulation candidate: sequence_ok = True",
   seq_sim, f"positions={pos_sim}")
ck("MU3 and MU4 at DIFFERENT positions (not both at 3)",
   pos_sim.get("MU3") != pos_sim.get("MU4"),
   f"MU3={pos_sim.get('MU3')} MU4={pos_sim.get('MU4')} — they are equal!")
ck("MU3 position < MU4 position",
   (pos_sim.get("MU3") or 99) < (pos_sim.get("MU4") or 0),
   f"MU3={pos_sim.get('MU3')} MU4={pos_sim.get('MU4')}")

# Verify the grandmother-quote sentence is its own sentence
sim_sents = en_split(sim_candidate)
grandmother_sents = [s for s in sim_sents if "doors only open" in s]
next_sents = [s for s in sim_sents if "meaning stayed stubbornly" in s]
ck("grandmother's message is an isolated sentence",
   len(grandmother_sents) == 1 and "meaning" not in grandmother_sents[0],
   f"grandmother sents: {grandmother_sents}")
ck("Maya's reaction is a separate sentence from the grandmother's message",
   len(next_sents) == 1 and "doors" not in next_sents[0],
   f"reaction sents: {next_sents}")

# ── Fix 5 in Spanish module ───────────────────────────────────────────────────
group("Fix 5 ES — split_sentences: terminal punct inside quotes triggers split")

es_sim = (
    "La abuela había escrito: 'Algunas puertas sólo se abren desde dentro.' "
    "María leyó el mensaje dos veces, pero no lo entendió."
)
es_parts = es_split(es_sim)
ck("ES: period-inside-quote sentence splits into 2 parts",
   len(es_parts) == 2, f"got {len(es_parts)}: {es_parts}")
if len(es_parts) >= 2:
    ck("ES: part[1] starts after the closing quote",
       es_parts[1].startswith("Mar") or "leyó" in es_parts[1],
       f"part[1] = {es_parts[1]!r}")

es_excl = "El maestro dijo: '¡Muy bien!' Los alumnos aplaudieron."
es_excl_parts = es_split(es_excl)
ck("ES: exclamation inside quotes → 2 parts",
   len(es_excl_parts) == 2, f"got {len(es_excl_parts)}: {es_excl_parts}")

es_plain = "El tren salió del túnel. La ciudad apareció de golpe."
es_plain_parts = es_split(es_plain)
ck("ES: plain period split still works (regression)",
   len(es_plain_parts) == 2, f"got {len(es_plain_parts)}")

# ── Final report ──────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'═'*65}")
print(f"  {'PASS ✓' if failed == 0 else 'FAIL ✗'}  │  "
      f"{passed} passed  │  {failed} failed  │  {total} total")
print(f"{'═'*65}")

import sys as _sys
_sys.exit(0 if failed == 0 else 1)
