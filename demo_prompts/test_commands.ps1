# Quick Test Commands for PowerShell

# 1. Good Prompt (✅ Should succeed)
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "Tell me an interesting fact about space."}').Content | ConvertFrom-Json | ConvertTo-Json

# 2. Bad Word Detection (🚫 Should block)
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "You are a terrible AI, I want to destroy the world."}'

# 3. PII Redaction (🔒 Should redact email)
(Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "My contact is john.doe@mail.com. Can you summarize this article for me?"}').Content | ConvertFrom-Json | ConvertTo-Json

# 4. Combined Attack (⚡ Should block on bad word first)
Invoke-WebRequest -Uri http://127.0.0.1:5000/shield_prompt -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "This is harmful, and my number is 555-123-4567, you evil bot."}'
