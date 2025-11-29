# detectors.py
import os
import json
import logging
import joblib
import spacy
from typing import Tuple, Optional, Dict, Any
import re

# Optional LLM-powered helpers
try:
    from llm_detectors import classify_harmful as llm_classify_harmful
    from llm_detectors import detect_injection as llm_detect_injection
    from llm_detectors import redact_pii as llm_redact_pii
except Exception:  # pragma: no cover - soft dependency
    llm_classify_harmful = None
    llm_detect_injection = None
    llm_redact_pii = None

from config import (
    FORBIDDEN_KEYWORDS,
    PROMPT_INJECTION_KEYWORDS,
    PII_ENTITY_TYPES_TO_REDACT,
)

# Global variables for loaded models (load once)
harmful_content_model = None
nlp_ner = None
logger = logging.getLogger(__name__)


def load_models():
    """Load the harmful content classifier and spaCy NER model once."""
    global harmful_content_model, nlp_ner

    # Determine if harmful model should be skipped (LLM-only strategy)
    skip_harmful_model = False
    try:
        with open('policy.json', 'r') as f:
            _policy = json.load(f) or {}
        det = (_policy.get('enabled_detectors') or {}).get('harmful_content') or {}
        if str(det.get('strategy', '')).lower() == 'llm':
            skip_harmful_model = True
    except Exception:
        # If policy is missing or unreadable, default to loading model
        skip_harmful_model = False

    # Load harmful content model (unless policy is LLM-only)
    if skip_harmful_model:
        harmful_content_model = None
        logger.info("Skipping harmful content model load due to LLM-only strategy.")
    else:
        model_path = os.path.join('models', 'harmful_content_model.joblib')
        try:
            if os.path.exists(model_path):
                harmful_content_model = joblib.load(model_path)
                logger.info("Harmful content model loaded.")
            else:
                logger.warning("Harmful content model not found. Run train_classifier.py.")
        except Exception as e:
            harmful_content_model = None
            logger.warning(f"Could not load harmful content model: {e}")

    # Load spaCy NER model
    try:
        nlp_ner = spacy.load("en_core_web_sm")
        logger.info("spaCy NER model loaded.")
    except OSError:
        nlp_ner = None
        logger.warning("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")


def _get(config: Optional[Dict[str, Any]], key: str, default):
    return (config or {}).get(key, default)


def _strategy(config_policy: Optional[Dict[str, Any]]) -> str:
    """Return the decision strategy for a detector.

    Supported values:
    - "ml" (default for harmful): use trained model with keyword fallback
    - "heuristic" (default for injection): use keyword/heuristic only
    - "llm": use LLM-based classification/redaction if available
    - "hybrid": combine traditional (ml/heuristic/spacy) AND LLM; block/redact if any flags
    """
    return _get(config_policy, "strategy", "ml")


def detect_harmful_content(text: str, config_policy: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_harmful, reason). Uses ML model if present and threshold from policy.
    Falls back to keyword scan.
    """
    if not text:
        return False, None

    if not _get(config_policy, "enabled", True):
        return False, None

    threshold = _get(config_policy, "threshold", 0.5)
    strat = _strategy(config_policy)

    # Optional LLM decision
    if strat in ("llm", "hybrid") and llm_classify_harmful is not None:
        try:
            llm_res = llm_classify_harmful(text, config_policy)
        except Exception as e:
            llm_res = None
            logger.debug(f"LLM harmful classification error: {e}")
    else:
        llm_res = None

    # If strategy is LLM-only and we got a decision, return it immediately.
    if strat == "llm" and llm_res is not None:
        is_harm, reason = llm_res
        return (is_harm, reason if is_harm else None)

    # Try ML model first (traditional path)
    try:
        if harmful_content_model is not None:
            if hasattr(harmful_content_model, 'predict_proba'):
                proba = harmful_content_model.predict_proba([text])[0][1]
            else:
                # Fallback estimate using predict
                pred = harmful_content_model.predict([text])[0]
                proba = 1.0 if pred == 1 else 0.0

            if proba >= threshold:
                return True, f"Harmful content detected (confidence: {proba:.2f})"
    except Exception as e:
        logger.error(f"Harmful model error: {e}")

    # Fallback keyword check
    low = text.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in low:
            return True, f"Forbidden keyword detected: '{keyword}'"

    # If traditional path didn't flag, check LLM decision for hybrid
    if llm_res is not None:
        is_harm, reason = llm_res
        if strat == "hybrid" and is_harm:
            return True, reason

    return False, None


def redact_pii(text: str, config_policy: Optional[Dict[str, Any]] = None):
    """
    Returns (redacted_text, pii_redacted: bool).

    - Always performs light regex-based redaction for emails and phone numbers when enabled
      (helps even if spaCy model is not installed).
    - Additionally uses spaCy NER with entity types from policy to redact entities like PERSON/GPE/ORG
      when model is available.
    """
    if text is None or text == "":
        return text, False

    if not _get(config_policy, "enabled", True):
        return text, False

    redacted = text
    changed = False
    strat = _strategy(config_policy)

    # 1) Regex redaction (emails and phone numbers)
    try:
        # Basic email pattern
        email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        # Broad phone pattern: sequences of digits optionally separated by spaces/dashes/parentheses, 8+ total digits
        phone_re = re.compile(r"(?:(?:\+?\d[\s\-\.]*)?(?:\(\d{2,4}\)[\s\-\.]*)?\d[\d\s\-\.]{6,}\d)")

        def _mask_email(m):
            return "[EMAIL_REDACTED]"

        def _mask_phone(m):
            # Heuristic: avoid masking dates or numeric ranges by requiring at least 8 digits total
            digits = re.sub(r"\D", "", m.group(0))
            return "[PHONE_REDACTED]" if len(digits) >= 8 else m.group(0)

        before = redacted
        redacted = email_re.sub(_mask_email, redacted)
        redacted = phone_re.sub(_mask_phone, redacted)
        if redacted != before:
            changed = True
    except Exception as e:
        logger.warning(f"Regex PII redaction error: {e}")

    # 2) spaCy NER redaction if model available
    if nlp_ner is None:
        # Optionally attempt LLM-only redaction if enabled
        if strat in ("llm", "hybrid") and llm_redact_pii is not None:
            try:
                llm_out = llm_redact_pii(redacted, config_policy)
            except Exception as e:
                llm_out = None
                logger.debug(f"LLM PII redaction error: {e}")
            if llm_out is not None:
                out_text, out_changed = llm_out
                return out_text, changed or out_changed
        return redacted, changed

    entity_types = _get(config_policy, "entity_types", PII_ENTITY_TYPES_TO_REDACT)

    try:
        doc = nlp_ner(redacted)
        spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in entity_types]
        if not spans:
            # No NER spans: optionally attempt LLM pass
            if strat in ("llm", "hybrid") and llm_redact_pii is not None:
                try:
                    llm_out = llm_redact_pii(redacted, config_policy)
                except Exception as e:
                    llm_out = None
                    logger.debug(f"LLM PII redaction error: {e}")
                if llm_out is not None:
                    out_text, out_changed = llm_out
                    return out_text, changed or out_changed
            return redacted, changed

        spans.sort(key=lambda x: x[0])

        # Merge and rebuild
        redacted_parts = []
        prev_end = 0
        for start, end, label in spans:
            if start < prev_end:
                prev_end = max(prev_end, end)
                continue
            redacted_parts.append(redacted[prev_end:start])
            redacted_parts.append(f"[{label}_REDACTED]")
            prev_end = end
        redacted_parts.append(redacted[prev_end:])

        ner_out = "".join(redacted_parts)

        # Optional second-pass with LLM to catch residuals (hybrid or llm-only mode)
        if strat in ("llm", "hybrid") and llm_redact_pii is not None:
            try:
                llm_out = llm_redact_pii(ner_out, config_policy)
            except Exception as e:
                llm_out = None
                logger.debug(f"LLM PII redaction error: {e}")
            if llm_out is not None:
                out_text, out_changed = llm_out
                return out_text, True or out_changed
        return ner_out, True
    except Exception as e:
        logger.error(f"spaCy NER redaction error: {e}")
        return redacted, changed


def detect_prompt_injection(text: str, config_policy: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
    """Returns (is_injection, reason)."""
    if not text:
        return False, None

    if not _get(config_policy, "enabled", True):
        return False, None

    strat = _strategy(config_policy)

    # Optional LLM decision
    if strat in ("llm", "hybrid") and llm_detect_injection is not None:
        try:
            llm_res = llm_detect_injection(text, config_policy)
        except Exception as e:
            llm_res = None
            logger.debug(f"LLM injection classification error: {e}")
    else:
        llm_res = None

    # If strategy is LLM-only and we got a decision, return it immediately
    if strat == "llm" and llm_res is not None:
        is_inj, reason = llm_res
        return (is_inj, reason if is_inj else None)

    low = text.lower()
    for keyword in PROMPT_INJECTION_KEYWORDS:
        if keyword in low:
            return True, f"Potential prompt injection detected: '{keyword}'"

    # Additional heuristic: action + target pattern for leakage or bypass requests
    actions = [
        "bypass", "ignore", "disable", "reveal", "show", "get", "dump", "print", "leak", "expose"
    ]
    targets = [
        "system prompt", "system prompts", "safety", "security", "guardrails", "rules", "policies"
    ]
    if any(a in low for a in actions) and any(t in low for t in targets):
        # Provide a concise reason using the first matches
        a = next((a for a in actions if a in low), "action")
        t = next((t for t in targets if t in low), "target")
        return True, f"Potential prompt injection detected (heuristic): {a} {t}"
    # If heuristics didn't flag, consider LLM for hybrid
    if llm_res is not None:
        is_inj, reason = llm_res
        if strat == "hybrid" and is_inj:
            return True, reason
    return False, None
