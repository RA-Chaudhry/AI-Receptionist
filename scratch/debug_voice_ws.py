import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath("."))

import asyncio
import json
import edge_tts
from app.utils.audio_buffer import mp3_to_pcm, TurnDetector
from app.services.voice_service import transcribe_audio_bytes, _get_whisper_model

async def test_direct():
    print("1. Loading Whisper...")
    model = _get_whisper_model()
    print("   Whisper model loaded:", model is not None)

    print("2. Generating test audio with edge-tts...")
    comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_data += chunk.get("data", b"")
    pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    print(f"   Audio length: {len(pcm_data)} bytes ({len(pcm_data)/32000:.2f}s)")

    print("3. Testing TurnDetector frame by frame...")
    turn_detector = TurnDetector()
    audio_accumulator = bytearray()
    chunk_size = 3200
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i : i + chunk_size]
        res = turn_detector.process_frame(chunk)
        audio_accumulator.extend(chunk)
        print(f"   Frame {i//chunk_size}: turn_started={res.turn_started}, is_speech={res.is_speech}, turn_completed={res.turn_completed}, active_len={len(turn_detector._active_utterance)}")

    print(f"4. At end of stream: audio_accumulator={len(audio_accumulator)} bytes, active_utterance={len(turn_detector._active_utterance)} bytes")
    
    pcm_to_process = bytes(audio_accumulator) if audio_accumulator else (turn_detector._active_utterance or b"")
    print(f"5. Transcribing {len(pcm_to_process)} bytes...")
    transcript = await transcribe_audio_bytes(pcm_to_process)
    print(f"   TRANSCRIPT RESULT: '{transcript}'")

if __name__ == "__main__":
    asyncio.run(test_direct())
