from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import re

from detectors import (
    load_models,
    detect_harmful_content,
    redact_pii,
    detect_prompt_injection,
)

# Load environment variables from .env file
load_dotenv()

# Configure the API key (you'll need to set this as an environment variable)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Serve assets from templates/public at /public and keep templates working
app = Flask(
    __name__,
    static_folder=os.path.join('templates', 'public'),
    static_url_path='/public'
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prompt_shield.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load policy (externalized configuration)
policy = {}
try:
    with open('policy.json', 'r') as f:
        policy = json.load(f)
    logger.info("Policy loaded successfully.")
except FileNotFoundError:
    logger.warning("policy.json not found. Using default empty policy.")

# Load models once at startup
load_models()


def log_event(event_type: str,
              detector: str | None = None,
              status: str | None = None,
              reason: str | None = None,
              original_prompt: str | None = None,
              processed_prompt: str | None = None,
              llm_response: str | None = None,
              metadata: dict | None = None):
    """Emit a structured JSON event to the log for easy parsing by /api/logs."""
    evt = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        "detector": detector,
        "status": status,
        "reason": reason,
        "original_prompt": original_prompt,
        "processed_prompt": processed_prompt,
        # Avoid overly large payloads in logs
        "llm_response_preview": (llm_response[:200] + ("…" if llm_response and len(llm_response) > 200 else "")) if llm_response else None,
        "metadata": metadata or {},
    }
    try:
        logger.info("EVENT_JSON " + json.dumps(evt, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Failed to log structured event: {e}")

@app.route('/')
def home():
    # Make Spline embed the default landing page
    return render_template('test.html')

@app.route('/demo')
def demo():
    # Redirect legacy demo route to the demo section on the main page
    return redirect(url_for('home') + '#demo')

@app.route('/features')
def features():
    # Redirect legacy features route to the features section on the main page
    return redirect(url_for('home') + '#features')


@app.route('/test')
def test_page():
    # Consolidate to main homepage
    return redirect(url_for('home'))


@app.route('/public/<path:filename>')
def public_assets(filename):
    """Serve assets from templates/public to use as a lightweight public folder."""
    assets_dir = os.path.join(app.root_path, 'templates', 'public')
    return send_from_directory(assets_dir, filename)

@app.route('/shield_prompt', methods=['POST'])
def shield_prompt():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Please provide a 'prompt' in the request body."}), 400

        user_prompt = data['prompt']
        logger.info(f"Received prompt: {user_prompt}")

        trace = []
        # --- Shielding Logic ---
        # Load per-detector policies with safe defaults
        detector_policies = (policy or {}).get("enabled_detectors", {})
        harmful_policy = detector_policies.get("harmful_content", {"enabled": True, "threshold": 0.5})
        pii_policy = detector_policies.get("pii_redaction", {"enabled": True})
        injection_policy = detector_policies.get("prompt_injection", {"enabled": True})

        # 1. Harmful Content Check
        is_harmful, harmful_reason = detect_harmful_content(user_prompt, harmful_policy)
        trace.append({
            "step": "harmful_content",
            "strategy": harmful_policy.get("strategy", "ml"),
            "decision": "block" if is_harmful else "allow",
            "reason": harmful_reason
        })
        if is_harmful:
            log_event(
                event_type="BLOCK",
                detector="harmful_content",
                status="blocked",
                reason=harmful_reason,
                original_prompt=user_prompt,
            )
            return jsonify({
                "status": "blocked",
                "reason": harmful_reason,
                "original_prompt": user_prompt,
                "trace": trace
            }), 403
        
        # 2. Prompt Injection / Jailbreak Heuristics
        is_injection, injection_reason = detect_prompt_injection(user_prompt, injection_policy)
        trace.append({
            "step": "prompt_injection",
            "strategy": injection_policy.get("strategy", "heuristic"),
            "decision": "block" if is_injection else "allow",
            "reason": injection_reason
        })
        if is_injection:
            log_event(
                event_type="BLOCK",
                detector="prompt_injection",
                status="blocked",
                reason=injection_reason,
                original_prompt=user_prompt,
            )
            return jsonify({
                "status": "blocked",
                "reason": injection_reason,
                "original_prompt": user_prompt,
                "trace": trace
            }), 403

        # 3. Advanced PII Detection and Redaction (spaCy NER)
        processed_prompt, pii_redacted = redact_pii(user_prompt, pii_policy)
        if pii_redacted:
            logger.info(f"Prompt after PII redaction: {processed_prompt}")
            log_event(
                event_type="REDACT",
                detector="pii_redaction",
                status="redacted",
                original_prompt=user_prompt,
                processed_prompt=processed_prompt,
            )
        trace.append({
            "step": "pii_redaction",
            "strategy": pii_policy.get("strategy", "ml"),
            "decision": "redacted" if pii_redacted else "unchanged",
            "meta": {"entity_types": pii_policy.get("entity_types")}
        })
        
        # --- Generate LLM response using Google Gemini ---
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(processed_prompt)
            llm_response = response.text
            trace.append({
                "step": "llm_generation",
                "model": "gemini-2.5-flash",
                "decision": "ok",
            })
        except Exception as e:
            llm_response = f"Error communicating with LLM: {str(e)}"
            logger.error(f"LLM Error: {e}")
            trace.append({
                "step": "llm_generation",
                "model": "gemini-2.5-flash",
                "decision": "error",
                "reason": str(e)
            })

        # Optional: Screen the LLM's response too
        response_policy = (policy or {}).get("response_screening", {})
        # Ensure we always have a final response variable
        final_llm_response = llm_response
        if response_policy.get("enabled", False):
            resp_det = response_policy.get("detectors", {})
            # Harmful content in response
            resp_harmful, resp_reason = detect_harmful_content(llm_response, resp_det.get("harmful_content", {"enabled": False}))
            trace.append({
                "step": "response_harmful_content",
                "strategy": (resp_det.get("harmful_content") or {}).get("strategy", "ml"),
                "decision": "block" if resp_harmful else "allow",
                "reason": resp_reason
            })
            if resp_harmful:
                log_event(
                    event_type="BLOCK",
                    detector="response_harmful_content",
                    status="blocked_response",
                    reason=resp_reason,
                    original_prompt=user_prompt,
                    processed_prompt=processed_prompt,
                    llm_response=llm_response,
                )
                return jsonify({
                    "status": "blocked_response",
                    "reason": resp_reason,
                    "original_prompt": user_prompt,
                    "llm_output_blocked": llm_response,
                    "trace": trace
                }), 403

            # PII in response
            final_llm_response, response_pii_redacted = redact_pii(llm_response, resp_det.get("pii_redaction", {"enabled": False}))
            if response_pii_redacted:
                logger.info(f"LLM response after PII redaction: {final_llm_response}")
                log_event(
                    event_type="REDACT",
                    detector="response_pii_redaction",
                    status="redacted_response",
                    original_prompt=user_prompt,
                    processed_prompt=processed_prompt,
                    llm_response=final_llm_response,
                )
            trace.append({
                "step": "response_pii_redaction",
                "strategy": (resp_det.get("pii_redaction") or {}).get("strategy", "ml"),
                "decision": "redacted" if response_pii_redacted else "unchanged"
            })
        else:
            final_llm_response = llm_response

        log_event(
            event_type="SUCCESS",
            detector=None,
            status="success",
            original_prompt=user_prompt,
            processed_prompt=processed_prompt,
            llm_response=final_llm_response,
        )
        return jsonify({
            "status": "success",
            "original_prompt": user_prompt,
            "processed_prompt": processed_prompt,
            "llm_response": final_llm_response,
            "trace": trace
        })
    except Exception as e:
        logger.exception(f"Unhandled error in /shield_prompt: {e}")
        return jsonify({
            "status": "error",
            "error": "Internal error while processing prompt.",
            "reason": str(e)
        }), 500


@app.route('/api/policy', methods=['GET'])
def get_policy():
    """Return the currently loaded policy JSON."""
    try:
        return jsonify(policy)
    except Exception as e:
        logger.error(f"Failed to return policy: {e}")
        return jsonify({"error": "Failed to load policy."}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Return recent structured events parsed from prompt_shield.log.

    Query params:
      - limit: number of recent lines to scan from the end of the file (default 500)
    """
    log_path = os.path.join(os.getcwd(), "prompt_shield.log")
    limit = request.args.get("limit", default=500, type=int)
    events = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Look back over the last N lines for EVENT_JSON entries
        for line in lines[-limit:]:
            idx = line.find("EVENT_JSON ")
            if idx != -1:
                payload = line[idx + len("EVENT_JSON "):].strip()
                try:
                    evt = json.loads(payload)
                    events.append(evt)
                except json.JSONDecodeError:
                    # ignore malformed structured events
                    continue
    except FileNotFoundError:
        logger.warning("prompt_shield.log not found when fetching logs.")
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return jsonify({"error": "Failed to read logs."}), 500

    return jsonify({"events": events})

if __name__ == '__main__':
    # Bind to the PORT environment variable for container platforms (defaults to 8080)
    port = int(os.environ.get("PORT", 8080))
    # Listen on all interfaces so Cloud Run can reach the container
    app.run(host="0.0.0.0", port=port, debug=False)