from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import re
import pandas as pd
import time
import sys

# Runtime check for Python 3.13 compatibility
if sys.version_info >= (3, 13):
    print("WARNING: Python 3.13+ detected. Some features (torch/transformers) may not work correctly.")
    print("Recommended: Python 3.10 or 3.11.")

from werkzeug.utils import secure_filename

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

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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

# Import vector_db for Hallucination Auditor (uses ChromaDB built-in embeddings)
try:
    import vector_db
    AUDITOR_AVAILABLE = True
    logger.info("Vector DB module loaded successfully - Hallucination Auditor available")
except KeyboardInterrupt:
    raise
except Exception as e:
    AUDITOR_AVAILABLE = False
    vector_db = None
    logger.error(f"Vector DB module failed to load: {str(e)[:200]}")
    logger.error("Hallucination Auditor features unavailable. Install with: pip install chromadb")

# Lazy import for Model Forge (uses transformers + torch which have Python 3.13 compatibility issues)
FORGE_AVAILABLE = False
run_fine_tuning = None

def _load_fine_tuner():
    global FORGE_AVAILABLE, run_fine_tuning
    if run_fine_tuning is None:
        try:
            from fine_tuner import run_fine_tuning as _run_fine_tuning
            run_fine_tuning = _run_fine_tuning
            FORGE_AVAILABLE = True
            logger.info("Fine tuner module loaded successfully - Model Forge available")
            return True
        except Exception as e:
            logger.error(f"Fine tuner module failed: {str(e)[:200]}")
            logger.error("Model Forge unavailable (Python 3.13 has compatibility issues with transformers/torch)")
            FORGE_AVAILABLE = False
            return False
    return FORGE_AVAILABLE

# Lazy import for Unlearning Engine (uses transformers + torch + peft)
UNLEARNING_AVAILABLE = False
run_unlearning = None

def _load_unlearner():
    global UNLEARNING_AVAILABLE, run_unlearning
    if run_unlearning is None:
        try:
            from unlearner import run_unlearning as _run_unlearning
            run_unlearning = _run_unlearning
            UNLEARNING_AVAILABLE = True
            logger.info("Unlearner module loaded successfully - Unlearning Engine available")
            return True
        except Exception as e:
            logger.error(f"Unlearner module failed: {str(e)[:200]}")
            logger.error("Unlearning unavailable (Python 3.13 has compatibility issues with transformers/torch)")
            UNLEARNING_AVAILABLE = False
            return False
    return UNLEARNING_AVAILABLE

# Note: Fine tuner and unlearner will be loaded on-demand when their endpoints are accessed
logger.info("Fine tuner and unlearner will be loaded on-demand (deferred due to potential Python 3.13 compatibility issues)")

from detectors import (
    load_models,
    detect_harmful_content,
    redact_pii,
    detect_prompt_injection,
)

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


@app.route('/unlearning')
def unlearning_page():
    """Renders the LLM Unlearning page."""
    return render_template('unlearning.html')


@app.route('/auditor')
def auditor_page():
    """Renders the Hallucination Auditor page."""
    return render_template('auditor.html')


@app.route('/api/auditor/status', methods=['GET'])
def api_auditor_status():
    """Check if the Hallucination Auditor is available."""
    return jsonify({
        "available": AUDITOR_AVAILABLE,
        "message": "Hallucination Auditor is ready" if AUDITOR_AVAILABLE else "Hallucination Auditor is not available on this deployment"
    })


@app.route('/forge')
def forge_page():
    """Renders the Model Forge page."""
    return render_template('forge.html')


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


import pandas as pd
from werkzeug.utils import secure_filename
# vector_db import removed - using module-level variable set at startup

# --- Obliviate Feature APIs ---

import random



@app.route('/api/unlearn', methods=['POST'])
def api_unlearn():
    """Performs unlearning on a model."""
    # Lazy load unlearner
    if not _load_unlearner():
        return jsonify({"error": "Unlearning Engine is not available. This feature requires Python 3.11 or compatible transformers/torch versions."}), 503
    
    if 'training_set' not in request.files or 'forget_set' not in request.files:
        return jsonify({"error": "Missing training or forget set file."}), 400
    
    training_file = request.files['training_set']
    forget_file = request.files['forget_set']
    info_to_forget = request.form.get('info_to_forget')

    if training_file.filename == '' or forget_file.filename == '' or not info_to_forget:
        return jsonify({"error": "Missing forget_set, training_set, or info_to_forget."}), 400

    training_path, forget_path = None, None
    try:
        # Save files temporarily
        training_filename = secure_filename(training_file.filename)
        training_path = os.path.join("uploads", training_filename)
        training_file.save(training_path)

        forget_filename = secure_filename(forget_file.filename)
        forget_path = os.path.join("uploads", forget_filename)
        forget_file.save(forget_path)

        # Determine model path
        model_path = "./forged_model"
        if not os.path.exists(model_path):
            model_path = "distilgpt2"
        
        logger.info(f"Starting unlearning for '{info_to_forget}' using model '{model_path}'")

        metrics = run_unlearning(
            model_path=model_path,
            forget_set_path=forget_path,
            info_to_forget=info_to_forget
        )

        logger.info("Unlearning process complete.")

        # Clean up temporary files
        os.remove(training_path)
        os.remove(forget_path)

        return jsonify({
            "status": "success",
            "retain_accuracy": metrics.get("retain_accuracy"),
            "forget_accuracy": metrics.get("forget_accuracy")
        })

    except Exception as e:
        logger.error(f"Error in /api/unlearn: {e}")
        # Clean up in case of error
        if training_path and os.path.exists(training_path):
            os.remove(training_path)
        if forget_path and os.path.exists(forget_path):
            os.remove(forget_path)
        return jsonify({"error": str(e)}), 500

@app.route('/api/auditor/upload', methods=['POST'])
def api_auditor_upload():
    """Handles dataset upload for the Hallucination Auditor."""
    if not AUDITOR_AVAILABLE:
        return jsonify({"error": "Hallucination Auditor is not available. Please use Python 3.11 or install required dependencies."}), 503
    
    if 'dataset' not in request.files:
        return jsonify({"error": "Missing dataset file."}), 400
    
    file = request.files['dataset']
    filename = secure_filename(file.filename)
    
    if filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Clear the existing collection before adding new documents
        vector_db.clear_collection()

        documents = []
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
            # Assuming the text is in a column named 'text'
            if 'text' in df.columns:
                documents = df['text'].tolist()
            elif 'content' in df.columns:
                documents = df['content'].tolist()
            else:
                return jsonify({"error": "CSV file must have a 'text' or 'content' column."}), 400
        elif filename.endswith('.json'):
            data = json.load(file)
            # Assuming the json is a list of strings
            if isinstance(data, list):
                documents = [str(item) for item in data]
            else:
                return jsonify({"error": "JSON file must be a list of strings."}), 400
        else:
            # Plain text file
            content = file.read().decode('utf-8')
            documents = content.split('\n')

        # Filter out empty documents
        documents = [doc for doc in documents if doc.strip()]
        
        if not documents:
            return jsonify({"error": "No documents found in the file."}), 400

        vector_db.add_documents_to_collection(documents)
        
        return jsonify({"status": "success", "message": f"{len(documents)} documents added to the vector store."})

    except Exception as e:
        logger.error(f"Error in /api/auditor/upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/auditor/query', methods=['POST'])
def api_auditor_query():
    """Queries the auditor's vector store and the LLM using ISR threshold."""
    if not AUDITOR_AVAILABLE:
        return jsonify({"error": "Hallucination Auditor is not available. Please use Python 3.11 or install required dependencies."}), 503
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query."}), 400
        
    query = data['query']
    custom_threshold = data.get('threshold')  # Optional custom threshold
    
    logger.info(f"Auditor query received: '{query}'")
    
    try:
        # Use the new ISR threshold checking mechanism
        isr_decision = vector_db.check_isr_threshold(query, custom_threshold)
        
        llm_response = "Query was blocked as it could not be verified against the provided dataset."
        
        # If decision allows, generate LLM response
        if isr_decision['allow']:
            logger.info(f"ISR check passed. Generating grounded LLM response.")
            
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                # Augment the prompt with the retrieved context
                retrieved_doc = isr_decision.get('matched_document', '')
                grounded_prompt = f"""
                You are a helpful assistant. Answer the following user query based *only* on the provided context document.
                If the context does not contain the answer, state that you cannot answer based on the provided information.

                CONTEXT:
                ---
                {retrieved_doc}
                ---
                USER QUERY: {query}
                """
                response = model.generate_content(grounded_prompt)
                llm_response = response.text
                logger.info(f"LLM generated response for query '{query}'")
                
            except Exception as e:
                llm_response = f"Error communicating with LLM: {str(e)}"
                logger.error(f"LLM Error during auditor query for '{query}': {e}")
        else:
            logger.info(f"ISR check failed. Query blocked. Reason: {isr_decision['reason']}")
        
        return jsonify({
            "status": "success",
            "decision": isr_decision['decision'],
            "isr_score": f"{isr_decision['isr_score']:.2f}",
            "threshold": f"{isr_decision['threshold']:.2f}",
            "confidence": isr_decision['confidence'],
            "explanation": isr_decision['explanation'],
            "reason": isr_decision['reason'],
            "llm_response": llm_response
        })

    except Exception as e:
        logger.exception(f"Error in /api/auditor/query for '{query}': {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/auditor/threshold', methods=['GET', 'POST'])
def api_auditor_threshold():
    """Get or set the ISR threshold configuration."""
    if not AUDITOR_AVAILABLE:
        return jsonify({"error": "Hallucination Auditor is not available. Please use Python 3.11 or install required dependencies."}), 503
    
    if request.method == 'GET':
        config = vector_db.get_isr_config()
        return jsonify(config)
    
    elif request.method == 'POST':
        data = request.json
        if not data or 'threshold' not in data:
            return jsonify({"error": "Missing threshold value."}), 400
        
        new_threshold = float(data['threshold'])
        success = vector_db.set_isr_threshold(new_threshold)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"ISR threshold updated to {new_threshold:.2f}",
                "config": vector_db.get_isr_config()
            })
        else:
            return jsonify({
                "error": "Invalid threshold value. Must be between min and max limits."
            }), 400


import time



@app.route('/api/forge/tune', methods=['POST'])
def api_forge_tune():
    """Fine-tunes a model using the provided dataset with enhanced tracking."""
    # Lazy load fine tuner
    if not _load_fine_tuner():
        return jsonify({"error": "Model Forge is not available. This feature requires Python 3.11 or compatible transformers/torch versions."}), 503
    
    if 'dataset' not in request.files:
        return jsonify({"error": "Missing dataset file."}), 400
    
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Get parameters from form data
        epochs = int(request.form.get('epochs', 1))
        learning_rate = float(request.form.get('learning_rate', 5e-5))

        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join("uploads", filename)
        file.save(temp_path)
        
        logger.info(f"Starting fine-tuning for {filename} with epochs={epochs}, lr={learning_rate}")

        # Run the fine-tuning process
        results = run_fine_tuning(
            dataset_path=temp_path,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Clean up the temporary file
        os.remove(temp_path)

        logger.info(f"Fine-tuning complete for {filename}.")

        return jsonify({
            "status": "success",
            "message": "Fine-tuning complete. Model ready for unlearning.",
            "model_path": results["model_path"],
            "loss_data": {
                "values": results["loss_history"],
                "steps": results["steps"],
                "final_loss": results["final_loss"]
            },
            "metadata": {
                "dataset_samples": results["dataset_samples"],
                "training_blocks": results["training_blocks"],
                "available_for_unlearning": True
            }
        })

    except Exception as e:
        logger.error(f"Error in /api/forge/tune: {e}")
        # Clean up in case of error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/list', methods=['GET'])
def api_models_list():
    """Lists available models (forged and unlearned) with their metadata."""
    try:
        models = []
        
        # Check for forged model
        if os.path.exists("./forged_model"):
            metadata_path = os.path.join("./forged_model", "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['type'] = 'forged'
                    models.append(metadata)
            else:
                models.append({
                    "type": "forged",
                    "output_path": "./forged_model",
                    "status": "available",
                    "available_for_unlearning": True
                })
        
        # Check for unlearned model
        if os.path.exists("./unlearned_model"):
            unlearned_metadata_path = os.path.join("./unlearned_model", "model_metadata.json")
            if os.path.exists(unlearned_metadata_path):
                with open(unlearned_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['type'] = 'unlearned'
                    models.append(metadata)
            else:
                models.append({
                    "type": "unlearned",
                    "output_path": "./unlearned_model",
                    "status": "available",
                    "available_for_unlearning": False
                })
        
        return jsonify({
            "status": "success",
            "models": models,
            "count": len(models)
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Bind to the PORT environment variable for container platforms (defaults to 8080)
    port = int(os.environ.get("PORT", 8080))
    # Listen on all interfaces so Cloud Run can reach the container
    app.run(host="0.0.0.0", port=port, debug=False)