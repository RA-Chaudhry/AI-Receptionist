# Real-Audio Test Samples (16kHz Mono 16-bit PCM WAV)

This directory holds offline pre-recorded test WAV audio files used by `scripts/e2e_voice_test.py` to benchmark real-time voice latency and VAD accuracy against a running AI Receptionist server without requiring a live physical microphone.

## Required Audio Specifications
- **Sample Rate:** 16,000 Hz (16 kHz)
- **Channels:** 1 (Mono)
- **Bit Depth:** 16-bit signed integer PCM
- **Format:** Standard RIFF/WAV

## Benchmark Phrases to Record

| File | Phrase / Intent | Category | Expected Response Path |
|------|-----------------|----------|------------------------|
| `00_background_noise.wav` | 3 seconds ambient room noise / silence | Noise Baseline | IDLE / No turn triggered |
| `01_hello.wav` | "Hello" | Greeting | Local Router / Canned cache (<600ms) |
| `02_office_timings.wav` | "What are your office timings?" | FAQ: Timings | Local Router / Canned cache (<600ms) |
| `03_office_location.wav` | "Where is your office?" | FAQ: Location | Local Router / Canned cache (<600ms) |
| `04_book_appointment.wav` | "I want to book an appointment." | Multi-turn booking | Router / Appointment Flow |
| `05_services.wav` | "What services do you offer?" | FAQ: Services | Local Router / Canned cache (<600ms) |
| `06_roman_urdu.wav` | "Aapki services kya hain?" | Roman Urdu FAQ | Local Router / LLM Urdu Stream |

## Generating / Updating WAV Samples
You can record your own voice using tools like Audacity / ffmpeg or generate synthetic audio using edge-tts or TTS tools converted to 16kHz mono WAV:
```bash
ffmpeg -i input_recording.m4a -ar 16000 -ac 1 -c:a pcm_s16le tests/audio_samples/01_hello.wav
```
