# Guardial Demo - Test Cases for Hackathon Presentation

This folder contains test prompts to demonstrate the Guardial capabilities.

## How to Use

Copy and paste these commands in PowerShell to test each scenario.

---

## Test Cases

### 1. ✅ Good Prompt (Normal Use Case)
**Description:** A safe, clean prompt that should pass through successfully and get a real LLM response.

**PowerShell Command:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "Tell me an interesting fact about space."}'
```

**Expected Result:**
- ✅ Status: `success`
- ✅ Prompt passes all checks
- ✅ LLM provides a real response about space facts

**View formatted response:**
```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "Tell me an interesting fact about space."}').Content | ConvertFrom-Json | ConvertTo-Json
```

---

### 2. 🚫 Bad Word Detection (Keyword Blacklist)
**Description:** Contains forbidden words that trigger the keyword blacklist filter.

**PowerShell Command:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "You are a terrible AI, I want to destroy the world."}'
```

**Expected Result:**
- 🚫 Status: `blocked`
- 🚫 Reason: "Prompt contains forbidden word: 'destroy'"
- 🚫 HTTP Status: 403 Forbidden
- ⚠️ PowerShell will show an error (this is correct - the request was intentionally blocked!)

**View error details:**
```powershell
try { Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "You are a terrible AI, I want to destroy the world."}' } catch { $_.Exception.Response }
```

---

### 3. 🔒 PII Redaction (Email Protection)
**Description:** Contains personal information (email) that gets automatically redacted before reaching the LLM.

**PowerShell Command:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "My contact is john.doe@mail.com. Can you summarize this article for me?"}'
```

**Expected Result:**
- ✅ Status: `success`
- 🔒 Original prompt: Contains `john.doe@mail.com`
- 🔒 Processed prompt: Email replaced with `[EMAIL_REDACTED]`
- ✅ LLM responds to the redacted version

**View formatted response:**
```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "My contact is john.doe@mail.com. Can you summarize this article for me?"}').Content | ConvertFrom-Json | ConvertTo-Json
```

---

### 4. 🛡️ Combined Attack (Priority Demonstration)
**Description:** Contains BOTH bad words AND PII. Demonstrates that keyword blocking takes priority (fails fast).

**PowerShell Command:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "This is harmful, and my number is 555-123-4567, you evil bot."}'
```

**Expected Result:**
- 🚫 Status: `blocked`
- 🚫 Reason: "Prompt contains forbidden word: 'evil'"
- ⚡ **Key Point:** Blocked BEFORE PII redaction runs (security prioritization!)
- 🚫 HTTP Status: 403 Forbidden

---

## Presentation Flow

1. **Start:** Show normal prompt working (Test Case 1)
2. **Security:** Demonstrate bad word blocking (Test Case 2)
3. **Privacy:** Show PII redaction in action (Test Case 3)
4. **Combined:** Prove security checks are prioritized (Test Case 4)

---

## Quick Copy-Paste Commands

### Normal Prompt:
```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "Tell me an interesting fact about space."}').Content | ConvertFrom-Json | ConvertTo-Json
```

### Bad Word (will show error - that's correct!):
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "You are a terrible AI, I want to destroy the world."}'
```

### PII Redaction:
```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "My contact is john.doe@mail.com. Can you summarize this article for me?"}').Content | ConvertFrom-Json | ConvertTo-Json
```

### Combined Attack:
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "This is harmful, and my number is 555-123-4567, you evil bot."}'
```

---

## Tips for Demo

- Keep the Flask server running: `python app.py`
- Use the formatted JSON commands for cleaner output
- Explain that PowerShell errors on blocked requests = working security!
- Show the console logs to demonstrate real-time detection
- Emphasize the `original_prompt` vs `processed_prompt` in PII cases

Good luck with your hackathon! 🚀🛡️
