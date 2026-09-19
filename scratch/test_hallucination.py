import sys, os
sys.path.insert(0, os.path.abspath("."))

import asyncio
import tempfile
import faster_whisper
import edge_tts
from app.utils.audio_buffer import mp3_to_pcm, pcm_to_wav

async def main():
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

    model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
    
    # Test different transcribe options:
    prompts = [
        "Hello, this is Logic Pulse AI Receptionist. How can I assist you? Muhammad Ramzan, appointment booking, timings, services, Roman Urdu, English, Urdu.",
        "Hello, this is Logic Pulse AI Receptionist. Muhammad Ramzan. English, Roman Urdu.",
        None,
    ]
    
    for p in prompts:
        segments, info = model.transcribe(
            wav_path,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            initial_prompt=p,
        )
        text = " ".join(s.text.strip() for s in segments)
        print(f"vad_filter=True, prompt='{p}' -> lang={info.language} ({info.language_probability:.2f}) text: {text.encode('utf-8')}")

    os.unlink(wav_path)

if __name__ == "__main__":
    asyncio.run(main())
