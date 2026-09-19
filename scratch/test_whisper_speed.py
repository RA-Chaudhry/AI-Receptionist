import sys, os
sys.path.insert(0, os.path.abspath("."))

import asyncio
import time
import os
import faster_whisper
import edge_tts
from app.utils.audio_buffer import mp3_to_pcm, pcm_to_wav
import tempfile

async def main():
    print(f"os.cpu_count() = {os.cpu_count()}")
    comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_data += chunk.get("data", b"")
    pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    wav_bytes = pcm_to_wav(pcm_data, sample_rate=16000, channels=1, sampwidth=2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name
        tmp.write(wav_bytes)

    # Test 1: threads=4 vs threads=os.cpu_count()
    for threads in [4, os.cpu_count() or 4]:
        for compute in ["int8"]:
            model = faster_whisper.WhisperModel("base", device="cpu", compute_type=compute, cpu_threads=threads)
            
            t0 = time.perf_counter()
            segments, info = model.transcribe(
                wav_path,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt="Hello, this is Logic Pulse AI Receptionist. How can I assist you? السلام علیکم! کیا میری آواز آ رہی ہے؟ Muhammad Ramzan, appointment booking, timings, services, Roman Urdu, English."
            )
            text = " ".join(s.text.strip() for s in segments)
            t1 = time.perf_counter()
            print(f"threads={threads}, compute={compute} -> {t1-t0:.3f}s: '{text}'")

    os.unlink(wav_path)

if __name__ == "__main__":
    asyncio.run(main())
