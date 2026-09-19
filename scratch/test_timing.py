import sys, os
sys.path.insert(0, os.path.abspath("."))

import asyncio
import time
import edge_tts
from app.services.voice_service import transcribe_audio_bytes, generate_tts_chunks
from app.services.ai_service import get_ai_response_stream
from app.utils.audio_buffer import mp3_to_pcm
from app.utils.prompts import voice_receptionist_prompt

async def main():
    print("Generating audio...")
    comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_data += chunk.get("data", b"")
    pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    
    # 1. Benchmark STT
    t0 = time.perf_counter()
    transcript = await transcribe_audio_bytes(pcm_data)
    t1 = time.perf_counter()
    print(f"[STT] Transcribe took {t1-t0:.3f}s -> '{transcript}'")
    
    # 2. Benchmark AI stream
    t0 = time.perf_counter()
    sentences = []
    async for sentence in get_ai_response_stream(
        user_name="guest",
        user_email="guest@example.com",
        message=transcript,
        history=[],
        system_prompt=voice_receptionist_prompt(),
    ):
        sentences.append(sentence)
    t1 = time.perf_counter()
    print(f"[LLM] AI stream took {t1-t0:.3f}s -> {sentences}")
    
    # 3. Benchmark TTS
    for sentence in sentences:
        t0 = time.perf_counter()
        audio_chunks = []
        async for chunk in generate_tts_chunks(sentence):
            audio_chunks.append(chunk)
        t1 = time.perf_counter()
        print(f"[TTS] generate_tts_chunks('{sentence}') took {t1-t0:.3f}s, {sum(len(c) for c in audio_chunks)} bytes")

if __name__ == "__main__":
    asyncio.run(main())
