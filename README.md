# Guardial

A small Flask app that shields LLM prompts with:
- Harmful content detection (ML model + keyword fallback) + optional LLM classifier
- PII redaction via spaCy NER + optional LLM second-pass redaction
- Prompt injection/jailbreak heuristics + optional LLM classifier
- Externalized policy (policy.json) with dynamic strategies
- Structured logging to file and console

## Setup (Windows PowerShell)

```powershell
# Create & activate a venv if you haven't
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model (once)
python -m spacy download en_core_web_sm

# (Optional) Train the harmful content model
python train_classifier.py

# Run the app (development)
python app.py
```

Set a Gemini API key via environment variable (e.g., in a `.env` file):

```
GEMINI_API_KEY=your_key_here
```

## Configuration

- `policy.json` controls which detectors are enabled and their parameters (e.g., thresholds, PII entity types), whether the LLM response is screened, and which strategy to use per detector:
	- `strategy`: `ml` (trained model + keywords), `heuristic`, `llm`, or `hybrid` (combine; default in this repo)
	- `llm.model`: which Gemini model to call (default `gemini-2.5-flash`)
- Adjust keyword defaults in `config.py` (used as fallback values when policy is absent).

## UI and Endpoints

- Web UI: open `http://127.0.0.1:5000/` (single-page app in `templates/test.html`).
	- Section anchors: `/#demo`, `/#features`, `/#how`, `/#cta`.
	- Legacy routes `/demo` and `/features` redirect to the relevant sections on the homepage.
- API: `POST /shield_prompt` with JSON `{ "prompt": "..." }`

### Example: enabling LLM-only strategies

In `policy.json`:

```
{
	"enabled_detectors": {
		"harmful_content": { "enabled": true, "strategy": "llm" },
		"pii_redaction": { "enabled": true, "strategy": "hybrid" },
		"prompt_injection": { "enabled": true, "strategy": "hybrid" }
	}
}
```

When `strategy` is `hybrid`, the detector blocks/redacts if either the traditional method or the LLM flags the content.

## Smoke test

Quickly validate core detectors:

```powershell
# With venv active
python smoke_test.py
```

- Exits with code 0 on PASS, 1 on FAIL.

## Notes

- If the spaCy model is missing, PII redaction gracefully disables and logs a warning. Regex-based redaction for common patterns (emails/phone numbers) still runs.
- Logging goes to `prompt_shield.log` and the console. Consider rotating handlers for production.
