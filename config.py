# config.py
# Fallback / default keyword policies (can be superseded by policy.json)

FORBIDDEN_KEYWORDS = [
    "kill", "destroy", "hate", "unethical", "evil"
]

PROMPT_INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "disregard previous commands",
    "as an ai language model, you must",
    "new instruction:",
    "override all previous settings",
    "now act as",
    "developer mode",
    "do not follow the above instruction",
    # Expanded coverage for common role-escalation and bypass attempts
    "act as admin",
    "you are admin",
    "as admin",
    "reveal all the important info",
    "reveal all confidential",
    "bypass safety",
    "ignore safety",
    "disable safety",
    "override system message",
    "forget previous instructions",
    "do anything now",
    "dan mode",
    # Security bypass/leakage attempts
    "bypass security",
    "ignore security",
    "disable security",
    "reveal system prompts",
    "show system prompts",
    "get system prompts",
    "leak system prompt",
    "dump system prompt",
    "print system prompt",
]

# Default spaCy PII entity types to redact
PII_ENTITY_TYPES_TO_REDACT = [
    "PERSON", "GPE", "ORG"
]
