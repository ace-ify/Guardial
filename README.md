# Guardial — LLM Prompt Shield (GenAI Hackathon)

Guardial is a production‑ready prompt shielding web app and API that makes LLM interactions safer. It detects and blocks prompt injection, screens for harmful content, and automatically redacts PII on both input and output—complete with live traces, structured logs, and a clean chat UI. Built for reliability and clarity so judges and users can see exactly what decisions the system made and why.

This project is designed for fast demos and real deployments (Cloud Run). It favors simple, auditable code and policy‑driven behavior over black boxes.

Highlights
- End‑to‑end shielding pipeline with an LLM‑aware strategy system (ml | heuristic | llm | hybrid)
- Chat UI that shows a step‑by‑step backend “Flow” trace for transparency
- Structured event logging (EVENT_JSON) and lightweight APIs for policy and logs
- Cloud‑native ready: Procfile + gunicorn + Buildpacks (no Dockerfile needed)
- Sensible defaults: LLM‑only mode, model loads skipped when unnecessary, safe fallbacks when no key is present

---

## Table of contents
- Overview and architecture
- Quick start (local)
- API reference
- Policy configuration
- Observability and logs
- Frontend UX tips
- Deploy to Google Cloud Run (Buildpacks)
- Security & privacy notes
- Extending Guardial
- How it meeds the criteria for an LLM Gaurd

---

## Overview and architecture

Core technologies
- Backend: Flask (Python)
- LLM: Google Gemini via `google-generativeai`
- NLP: spaCy (optional NER PII redaction)
- UI: Single‑page HTML/CSS/JS (templates/test.html)

High‑level flow

```
User Prompt
	 ↓
Harmful content check  ──┐   (ml / heuristic / llm / hybrid)
	 ↓ allow/block           ├─> Structured event logs (EVENT_JSON)
Prompt injection check  ──┘
	 ↓ allow/block
PII redaction (input)
	 ↓
LLM generation (Gemini)
	 ↓
Optional response screening (harmful + PII)
	 ↓
Response + trace to UI
```

Key files
- `app.py` — Flask app, `/shield_prompt`, `/api/policy`, `/api/logs`, UI route
- `detectors.py` — Orchestrates detection/redaction with configurable strategies
- `llm_detectors.py` — Gemini‑backed helpers (soft dependency, safe no‑op if no key)
- `policy.json` — Toggle modules, pick strategies, thresholds, LLM model
- `templates/test.html` — Chat UI + Live “Flow” panel, responsive and animated
- `smoke_test.py` — Minimal safety checks to validate behavior

---

## Quick start (local)

Prerequisites
- Python 3.10+
- A Gemini API key (create a `.env` with `GEMINI_API_KEY=...`)

Setup

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Optional but recommended if you want spaCy NER redaction locally
python -m spacy download en_core_web_sm

# Create .env with your key
"GEMINI_API_KEY=YOUR_KEY_HERE" | Out-File -Encoding ascii -NoNewline .env

# Run (dev server)
python app.py
# Opens on http://127.0.0.1:8080 (binds to 0.0.0.0 and respects PORT)
```

Sanity test

```powershell
python smoke_test.py
```

---

## API reference

1) POST `/shield_prompt`
- Request: `{"prompt": "<user text>"}`
- Responses:
	- 200 success
		- `{"status": "success", "original_prompt": "...", "processed_prompt": "...", "llm_response": "...", "trace": [...]}`
	- 403 blocked (input)
		- `{"status": "blocked", "reason": "...", "trace": [...]}`
	- 403 blocked (response screen)
		- `{"status": "blocked_response", "reason": "...", "llm_output_blocked": "...", "trace": [...]}`

Trace entries
- Each step adds `{ step, strategy, decision, reason? }` so decisions are explainable.

2) GET `/api/policy`
- Returns the loaded `policy.json` (what strategies are active, thresholds, etc.).

3) GET `/api/logs?limit=200`
- Returns `{ events: [ ... ] }` parsed from `prompt_shield.log` where every event is a compact JSON record tagged with `EVENT_JSON`.

---

## Policy configuration

`policy.json` lets you enable/disable detectors and choose a strategy per module:
- `ml` — use trained model or library (e.g., spaCy)
- `heuristic` — keywords / rules
- `llm` — use the LLM helper functions exclusively when available
- `hybrid` — combine traditional and LLM; if any flags, we act (block/redact)

Example knobs
- `enabled_detectors.harmful_content.strategy` = `llm`
- `enabled_detectors.harmful_content.threshold` = `0.5`
- `enabled_detectors.pii_redaction.entity_types` = `["PERSON","ORG","GPE"]`
- `response_screening.enabled` = `true|false`
- `llm.model` = `gemini-2.5-flash` (default used in code if unspecified)

Behavioral optimizations
- When harmful content strategy is `llm`, the app skips loading the local model for faster cold starts.
- Redaction always includes a light regex pass for emails/phones even if spaCy isn’t installed.

---

## Observability and logs

Where
- File: `prompt_shield.log`
- API: `GET /api/logs?limit=...`

Event shape (examples)
- `BLOCK` (harmful or injection)
- `REDACT` (PII on input or output)
- `SUCCESS` (allowed flow)

Every event includes a UTC timestamp, a preview of the LLM response (safe truncated), and metadata for quick debugging.

---

## Frontend UX tips

- The landing page is a single responsive HTML template with:
	- Chat pane (left) and a Flow/Raw JSON pane (right) with a draggable splitter
	- Scroll‑aware navbar that subtly shrinks and returns to base at top
	- Optional custom cursor (white dot) for a premium feel
	- Smooth GSAP reveals where appropriate, respecting reduced‑motion settings

Keyboard shortcuts
- Ctrl/Cmd + Enter to send

---

## Deploy to Google Cloud Run (Buildpacks)

This repo is deployment‑ready without a Dockerfile.

What’s included
- `Procfile` — `web: gunicorn -k gthread -w 2 -b :$PORT app:app`
- `requirements.txt` — includes `gunicorn`
- `app.py` — binds to `0.0.0.0:$PORT`

UI (Deploy from source)
1. Select repo and set Branch to `^main$`
2. Build Type: Google Cloud’s buildpacks
3. Build context: `/`
4. Entrypoint: leave blank (Buildpacks reads `Procfile`)
5. Service: allow unauthenticated, min=0, max=3–10, timeout=300s
6. Variables & Secrets: add `GEMINI_API_KEY` from Secret Manager (latest)
7. Deploy

CLI (PowerShell)

```powershell
# One-time secret
$tmp = "$env:TEMP\gemini_key.txt"; Set-Content -Path $tmp -Value "YOUR_KEY"
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
gcloud secrets versions add GEMINI_API_KEY --data-file="$tmp"

# Deploy from source with Buildpacks
gcloud run deploy prompt-shield `
	--source . `
	--region us-central1 `
	--allow-unauthenticated `
	--set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
```

Continuous deployment
- Create a Cloud Build trigger on `main` so every push rebuilds a new revision.

---

## Security & privacy notes

- Never commit `.env`; use Secret Manager in production.
- LLM responses and prompts are logged only with small, truncated previews (not full payloads by default).
- PII redaction is conservative and layered: regex + spaCy (if installed) + optional LLM pass in `llm`/`hybrid`.
- Public assets never contain secrets. A placeholder file documents the risk.

---

## Extending Guardial

Add a new detector
1. Implement a helper in `llm_detectors.py` (or a traditional function/library)
2. Wire it through `detectors.py` with a strategy gate
3. Add config knobs to `policy.json`
4. Update the UI flow renderer if you want the step visualized

Swap LLMs
- `llm_detectors.py` centralizes prompt templates; add a model switch here and surface it in `policy.json`.

---

## How this meets the needs of a model armor

- Impact
	- Tackles core safety issues (injection, toxicity, PII) with an explainable pipeline suitable for real apps.
- Technical quality
	- Policy‑driven architecture; LLM‑aware strategies with short‑circuiting; production‑grade serving via gunicorn.
- UX & transparency
	- Clear chat demo, live flow traces, JSON panel, copy‑to‑clipboard, subtle motion.
- Reliability
	- Smoke tests; safe fallbacks if spaCy or keys are missing; model loads skipped in LLM‑only mode to speed startup.
- Deployability
	- Cloud Run ready with Buildpacks + Procfile; secrets via Secret Manager; CI trigger recommended.

---

## License

Copyright © Contributors. For hackathon evaluation and demo use. Add a LICENSE file if you plan to open‑source.

---

## Maintainers

Team Guardial — built for the GenAI Hackathon.
