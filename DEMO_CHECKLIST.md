# 🎙️ Logic Pulse AI Receptionist — Live Demo Checklist & Runbook

This guide details the step-by-step procedure for preparing and conducting flawless, low-latency live demonstrations of the AI Receptionist.

---

## 1. Environment & Hardware Preparation

### ⚠️ Critical Requirement: Wear Headphones / Headset
- **Always use headphones or an external headset with a microphone.**
- Built-in laptop speakers cause acoustic feedback into the microphone, which can trigger accidental barge-in interruptions.

### Required Environment Variables
Ensure your `.env` or terminal session has the following configured:
```bash
# Enable Demo Mode (Enables latency logging, filler clips, conservative VAD, suppresses real SMS)
DEMO_MODE=true

# Groq Cloud API Key
GROQ_API_KEY=gsk_your_groq_api_key_here

# Fast Inference Models
GROQ_MODEL=llama-3.1-8b-instant
GROQ_WHISPER_MODEL=whisper-large-v3-turbo

# STT Provider & TTS Voice
STT_PROVIDER=groq_whisper
TTS_VOICE=en-US-JennyNeural

# Telemetry Output
TELEMETRY_LOG_FILE=logs/telemetry.jsonl
```

---

## 2. Server Startup & Verification

### Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Check Voice Stack Health
Open your browser or run curl to inspect:
```bash
curl http://localhost:8000/api/v1/health/voice
```
Verify the JSON response returns:
- `"status": "healthy"`
- `"groq_reachable": true`
- `"caches_warm": { "web_tts_cache_count": > 15, "vonage_pcm_cache_count": > 15 }`
- `"demo_mode": true`

---

## 3. Warm-Up Protocol (3 Practice Turns)

Before presenting to an audience, run through **3 quick warm-up turns** in the browser frontend (`http://localhost:8000/`):

1. **Turn 1 (Greeting & VAD sync):** Say `"Hello"`
   - *Verifies:* Audio buffer initialization, microphone permission, and prewarmed greeting cache.
2. **Turn 2 (Local Router Cache Hit):** Say `"Where is your office?"`
   - *Verifies:* Local rule engine routing (< 600ms latency, 0 Groq token cost).
3. **Turn 3 (LLM Connection & TLS Keepalive):** Say `"Tell me about your technology stack"`
   - *Verifies:* Groq streaming connection, token chunking, and Edge-TTS sentence pipeline.

---

## 4. Live Demo Script Order

Follow this structured narrative during the presentation to showcase each architectural strength:

### Phase 1: Instant Local Rule Routing ($0 Cost, < 600ms)
*Demonstrate instantaneous responses powered by pre-rendered caching.*
1. **Prompt:** `"What are your office timings?"`
   - *Expected:* Instant response: *"Our office hours are Monday through Friday, 9:00 AM to 5:00 PM."*
2. **Prompt:** `"Where is your office located?"`
   - *Expected:* Instant location reply with SMS offer.

### Phase 2: Natural Language LLM Streaming & Filler Clip
*Demonstrate dynamic conversational reasoning and seamless filler transitions.*
1. **Prompt:** `"Can you build a custom ERP system integrated with artificial intelligence for my hospital?"`
   - *Expected:* Filler clip (*"Ji,"* / *"Sure,"*) plays if LLM takes > 700ms, followed immediately by smooth clause-by-clause stream.
2. **Prompt (Roman Urdu):** `"Aapki development team kahan based hai aur kya technologies use karte hain?"`
   - *Expected:* Seamless Urdu voice model (`ur-PK-UzmaNeural`) response in Roman Urdu.

### Phase 3: Multi-Turn Appointment Booking Flow
*Demonstrate in-memory conversation state, digit normalization, and deterministic tool execution.*
1. **Turn 1:** `"I want to book an appointment."`
   - *Bot:* *"I can help you schedule that. Which day works best for you?"*
2. **Turn 2:** `"Friday."`
   - *Bot:* *"What time on Friday would you prefer?"*
3. **Turn 3:** `"2 PM."`
   - *Bot:* *"Please provide your phone number so I can confirm the reservation."*
4. **Turn 4:** Spoken digits: `"0 3 0 0 1 2 3 4 5 6 7"`
   - *Bot:* *"I have an appointment for Friday at 2 PM for 03001234567. Should I confirm this booking?"*
5. **Turn 5:** `"Yes, confirm."` (or Urdu: *"جی ہاں"*)
   - *Bot:* *"Your appointment is confirmed for Friday at 2 PM. A confirmation message has been recorded."*

### Phase 4: Full-Duplex Audio Barge-In
*Demonstrate natural human interruption.*
1. Ask a long question: `"Explain the complete history and architecture of Logic Pulse."`
2. While the bot is speaking its first sentence, interrupt firmly: `"Wait, tell me your phone number instead."`
3. *Expected:* Assistant immediately silences playback, cancels background TTS, and answers the new query.

---

## 5. Troubleshooting & Latency Spike Mitigations

| Issue | Root Cause | Immediate Action |
|-------|------------|-------------------|
| **Latency spike (> 3.5s on LLM turns)** | Groq Cloud region queue delay | Change model in `.env`: `GROQ_MODEL=llama-3.1-8b-instant` and restart server. |
| **Microphone not capturing audio** | Browser mic permissions blocked | Click padlock icon in browser address bar $\to$ Allow Microphone $\to$ Refresh page. |
| **Accidental barge-in during bot speech** | Acoustic feedback from laptop speakers | Lower speaker volume or wear headphones/earphones. |
| **STT fails on digits** | Mumbled spoken digits | Speak digits individually: *"zero three zero zero..."*. |
| **Server disconnects** | Keepalive timeout after 45s idle | Refresh web page or speak to resume WebSocket session. |
