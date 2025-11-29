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
from werkzeug.utils import secure_filename

# Runtime check for Python 3.13 compatibility
if sys.version_info >= (3, 13):
    print("WARNING: Python 3.13+ detected. Some features (torch/transformers) may not work correctly.")
    print("Recommended: Python 3.10 or 3.11.")

# Load environment variables from .env file
load_dotenv()

# Configure the API key
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

# Import vector_db for Hallucination Auditor
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

# Lazy import for Unlearning Engine
UNLEARNING_AVAILABLE = False
run_unlearning = None
generate_text = None

def _load_unlearner():
    global UNLEARNING_AVAILABLE, run_unlearning, generate_text
    if run_unlearning is None:
        try:
            from unlearner import run_unlearning as _run_unlearning, generate_text as _generate_text
            run_unlearning = _run_unlearning
            generate_text = _generate_text
            UNLEARNING_AVAILABLE = True
            logger.info("Unlearner module loaded successfully - Unlearning Engine available")
            return True
        except Exception as e:
            logger.error(f"Unlearner module failed: {str(e)[:200]}")
            logger.error("Unlearning unavailable (Python 3.13 has compatibility issues with transformers/torch)")
            UNLEARNING_AVAILABLE = False
            return False
    return UNLEARNING_AVAILABLE

# Lazy import for Model Forge
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

@app.route('/api/auditor/upload', methods=['POST'])
def api_auditor_upload():
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


@app.route('/unlearning')
def unlearning_page():
    return render_template('unlearning.html')


@app.route('/api/unlearn', methods=['POST'])
def api_unlearn():
    """Handles the unlearning process."""
    if not _load_unlearner():
        return jsonify({"error": "Unlearning Engine is not available. Please use Python 3.10/3.11."}), 503

    try:
        # Check for required files
        if 'training_set' not in request.files or 'forget_set' not in request.files:
            return jsonify({"error": "Missing training_set or forget_set file."}), 400
        
        training_file = request.files['training_set']
        forget_file = request.files['forget_set']
        info_to_forget = request.form.get('info_to_forget')

        if not info_to_forget:
            return jsonify({"error": "Missing info_to_forget."}), 400

        # Save files temporarily
        os.makedirs("uploads", exist_ok=True)
        training_path = os.path.join("uploads", secure_filename(training_file.filename))
        forget_path = os.path.join("uploads", secure_filename(forget_file.filename))
        
        training_file.save(training_path)
        forget_file.save(forget_path)

        # Determine model path (use forged model if available, else base)
        model_path = "./forged_model" if os.path.exists("./forged_model") else "distilgpt2"
        
        # Run unlearning
        results = run_unlearning(
            model_path=model_path,
            forget_set_path=forget_path,
            info_to_forget=info_to_forget
        )

        # Cleanup
        if os.path.exists(training_path): os.remove(training_path)
        if os.path.exists(forget_path): os.remove(forget_path)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in /api/unlearn: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/query_model', methods=['POST'])
def api_query_model():
    """Queries a specific model (base/forged or unlearned)."""
    if not _load_unlearner():
        return jsonify({"error": "Unlearning Engine is not available."}), 503

    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt."}), 400
    
    prompt = data['prompt']
    model_type = data.get('model_type', 'base') # 'base' or 'unlearned'
    
    try:
        # Determine model path
        if model_type == 'unlearned':
            model_path = "./unlearned_model"
            if not os.path.exists(model_path):
                return jsonify({"error": "Unlearned model not found. Please run unlearning first."}), 404
        else:
            # Base model (or forged if available)
            model_path = "./forged_model" if os.path.exists("./forged_model") else "distilgpt2"
            
        response_text = generate_text(model_path, prompt)
        
        return jsonify({
            "status": "success",
            "model": model_path,
            "response": response_text
        })

    except Exception as e:
        logger.error(f"Error in /api/query_model: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Bind to the PORT environment variable for container platforms (defaults to 8080)
    port = int(os.environ.get("PORT", 8080))
    # Listen on all interfaces so Cloud Run can reach the container
    app.run(host="0.0.0.0", port=port, debug=False)