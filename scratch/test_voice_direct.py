import sys, os
sys.path.insert(0, os.path.abspath("."))

import asyncio
import edge_tts
from app.services.voice_service import transcribe_audio_bytes, _get_whisper_model
from app.utils.audio_buffer import mp3_to_pcm

async def main():
    print("Testing _get_whisper_model()...")
    model = _get_whisper_model()
    print("Model:", model)
    
    print("Generating TTS audio...")
    comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_data += chunk.get("data", b"")
    pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    print(f"Generated {len(pcm_data)} PCM bytes ({len(pcm_data)/32000:.2f}s)")
    
    result = await transcribe_audio_bytes(pcm_data)
    print(f"Transcription result: '{result}'")

if __name__ == "__main__":
    asyncio.run(main())
