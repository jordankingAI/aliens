# Security Policy

## Scope

This document describes known security considerations for the ALIENS codebase. ALIENS is released under the MIT License as-is, with no warranties and no commitment to ongoing maintenance or updates. See [LICENSE](LICENSE).

---

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities by emailing the maintainer. Include:

- A description of the vulnerability and its potential impact.
- The affected module(s) and function(s).
- A minimal reproduction case if possible.
- Whether you believe this is exploitable in a deployed educational setting.

---

## Threat model

ALIENS is a backend engine. It does not serve HTTP requests, open network connections, or persist data. The attack surface is accordingly narrow, but the following categories are relevant:

### LLM output injection

ALIENS passes LLM responses through strict JSON contract validators before parsing. A malformed or adversarially crafted LLM response cannot cause arbitrary code execution — it will raise a `ValidationError` that the caller must handle. However:

- If a caller's `LLMBackend.complete_json()` implementation deserialises responses unsafely (e.g. using `eval()` rather than `json.loads()`), that vulnerability is in the caller's code, not in ALIENS.
- The `PromptLibrary` system prompts include the learner state and source passage text. If the source passage text originates from untrusted user input, prompt injection attacks against the LLM are the caller's responsibility to mitigate.

### Learner state manipulation

`LearnerState` is a frozen dataclass. It cannot be mutated in place. All state updates return new instances. There is no authentication or access control — the caller is responsible for ensuring that the correct `LearnerState` is passed for the correct learner. A caller that allows learners to supply their own state could allow band inflation or false disability flags.

**The `decoding_disability` flag in `DyslexicLearnerState` is especially sensitive.** A learner who falsely sets `decoding_disability = True` would receive signal adjustments that inflate their effective fluency and suppress hint counts. This could mask genuine comprehension difficulty. Callers must ensure this flag is set only by authorised practitioners.

### Denial of service

ALIENS makes exactly 6 LLM calls per cycle. There is no retry loop with unbounded attempts. A malicious source passage (e.g. extremely long text) would increase token consumption in LLM calls but cannot cause an infinite loop within ALIENS itself. Callers should apply their own rate limiting and input length validation before passing text to the engine.

### Sensitive data in logs

`prepare_cycle()` logs the passage ID and learner ID in DEGRADED mode warnings. It does not log passage text, learner retell content, or assessment answers. Callers should ensure that learner IDs used are pseudonymous rather than personally identifiable where required by applicable data protection law (e.g. FERPA, GDPR, COPPA).

---

## Dependencies

ALIENS has **no external runtime dependencies**. Standard library only. There is no supply chain attack surface from third-party packages.

---

## Educational context note

ALIENS is used in educational settings with minors. Contributors and deployers should be aware that:

- Systematic misclassification of a learner's ability — whether through a bug, a misconfigured engine, or improper use of the dyslexia flag — has direct educational consequences for a real child.
- Security vulnerabilities that allow band manipulation, false diagnostic labels, or suppression of genuine comprehension signals are considered high severity regardless of the technical exploit path.
- Any vulnerability that could allow an adult to monitor or extract a child's individual performance data without authorisation should be treated as critical.
