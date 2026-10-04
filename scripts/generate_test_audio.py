"""
scripts/generate_test_audio.py — Generates 16kHz mono WAV test samples for e2e benchmark.
"""

import asyncio
import io
import os
import sys
import wave
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import edge_tts
except ImportError:
    edge_tts = None

from app.utils.audio_buffer import mp3_to_pcm, pcm_to_wav

SAMPLES = [
    ("01_hello.wav", "Hello"),
    ("02_office_timings.wav", "What are your office timings?"),
    ("03_office_location.wav", "Where is your office?"),
    ("04_book_appointment.wav", "I want to book an appointment"),
    ("05_services.wav", "What services do you offer?"),
    ("06_roman_urdu.wav", "Aapki services kya hain?"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "audio_samples")


async def generate_samples():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 00_background_noise.wav (3 seconds of quiet room noise RMS ~30)
    noise_path = os.path.join(OUT_DIR, "00_background_noise.wav")
    num_samples = 16000 * 3
    noise = np.random.normal(0, 30, num_samples).astype(np.int16)
    with wave.open(noise_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(noise.tobytes())
    print(f"Generated: {noise_path}")

    # 2. Benchmark phrases
    for filename, text in SAMPLES:
        out_path = os.path.join(OUT_DIR, filename)
        if edge_tts is not None:
            try:
                voice = "ur-PK-UzmaNeural" if "Aapki" in text else "en-US-JennyNeural"
                communicate = edge_tts.Communicate(text, voice=voice)
                chunks = []
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        chunks.append(chunk["data"])
                mp3_bytes = b"".join(chunks)
                pcm = mp3_to_pcm(mp3_bytes, 16000)
                wav_bytes = pcm_to_wav(pcm, 16000)
                with open(out_path, "wb") as f:
                    f.write(wav_bytes)
                print(f"Generated TTS: {out_path} ({len(wav_bytes)} bytes)")
                continue
            except Exception as exc:
                print(f"Failed edge_tts for {filename}: {exc}")

        # Fallback synthetic tone
        t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(tone.tobytes())
        print(f"Generated synthetic tone: {out_path}")


if __name__ == "__main__":
    asyncio.run(generate_samples())
