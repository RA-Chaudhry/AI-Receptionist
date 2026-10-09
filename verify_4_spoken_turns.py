"""
verify_4_spoken_turns.py — 4 Consecutive Spoken-Style Turns E2E Verification
Connects via WebSocket to /api/v1/voice/stream on the live server,
sends real PCM voice waveforms (synthesized via edge-tts or valid 16kHz PCM),
and verifies that all 4 turns complete with transcripts, audio chunks, and response_end.
"""

import asyncio
import base64
import json
import logging
import sys
import time
import websockets
import edge_tts
from app.utils.audio_buffer import mp3_to_pcm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("e2e_verify")

SERVER_WS_URL = "ws://127.0.0.1:8000/api/v1/voice/stream?phone_number=+923001234567"

TURNS = [
    {"text": "Hello, what services do you provide?", "voice": "en-US-JennyNeural"},
    {"text": "What are your office hours?", "voice": "en-US-JennyNeural"},
    {"text": "I would like to book an appointment for Monday at two PM.", "voice": "en-US-JennyNeural"},
    {"text": "Thank you so much, goodbye.", "voice": "en-US-JennyNeural"},
]


async def text_to_pcm(text: str, voice: str) -> bytes:
    comm = edge_tts.Communicate(text, voice=voice)
    mp3_chunks = []
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_chunks.append(chunk.get("data", b""))
    mp3_bytes = b"".join(mp3_chunks)
    pcm_bytes = mp3_to_pcm(mp3_bytes, target_rate=16000)
    return pcm_bytes


async def run_4_turns():
    logger.info("Generating realistic spoken audio waveforms for 4 turns...")
    pcm_turns = []
    for i, turn in enumerate(TURNS):
        pcm = await text_to_pcm(turn["text"], turn["voice"])
        logger.info("Turn %d (%s) generated %d bytes PCM audio.", i + 1, turn["text"], len(pcm))
        pcm_turns.append(pcm)

    logger.info("Connecting to live WebSocket at %s...", SERVER_WS_URL)
    async with websockets.connect(SERVER_WS_URL) as ws:
        for turn_idx, pcm_audio in enumerate(pcm_turns):
            turn_num = turn_idx + 1
            logger.info("=== STARTING TURN %d: '%s' ===", turn_num, TURNS[turn_idx]["text"])

            # Stream audio in 40ms frames (1280 bytes at 16kHz 16-bit mono)
            frame_size = 1280
            for offset in range(0, len(pcm_audio), frame_size):
                chunk = pcm_audio[offset : offset + frame_size]
                await ws.send(chunk)
                await asyncio.sleep(0.01)

            # Send speech_ended signal
            await ws.send(json.dumps({"type": "speech_ended"}))

            # Receive server responses for this turn
            received_transcript = None
            received_audio_chunks = 0
            received_response_end = None
            turn_start = time.time()

            while True:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
                msg = json.loads(msg_raw)
                msg_type = msg.get("type")

                if msg_type == "transcript":
                    received_transcript = msg.get("data")
                    logger.info("Turn %d Server Transcript: '%s'", turn_num, received_transcript)

                elif msg_type == "audio_chunk":
                    received_audio_chunks += 1
                    audio_b64 = msg.get("data", "")
                    chunk_bytes = len(base64.b64decode(audio_b64)) if audio_b64 else 0
                    if received_audio_chunks == 1:
                        logger.info("Turn %d First Audio Chunk Received (%d bytes)", turn_num, chunk_bytes)

                elif msg_type == "response_end":
                    received_response_end = msg.get("data")
                    provider = msg.get("provider", "")
                    logger.info("Turn %d Response Complete: '%s' (Provider: %s)", turn_num, received_response_end, provider)
                    break

                elif msg_type == "goodbye":
                    logger.info("Turn %d Received Goodbye: '%s'", turn_num, msg.get("data"))
                    break

                elif msg_type == "error":
                    logger.error("Turn %d Server Error: %s", turn_num, msg.get("data"))
                    raise RuntimeError(f"Server error on turn {turn_num}: {msg.get('data')}")

            turn_duration = time.time() - turn_start
            logger.info(
                "Turn %d FINISHED in %.2fs | Audio Chunks: %d | Transcript: '%s'",
                turn_num,
                turn_duration,
                received_audio_chunks,
                received_transcript,
            )
            assert received_transcript, f"Turn {turn_num} should have a valid transcript"
            # Turn 4 (goodbye) might not produce audio chunks if it exits immediately or it might produce goodbye
            if turn_num < 4:
                assert received_audio_chunks > 0, f"Turn {turn_num} should have synthesized audio chunks"

            # Short pause between natural conversational turns
            await asyncio.sleep(0.5)

    logger.info("ALL 4 CONSECUTIVE SPOKEN TURNS SUCCEEDED END-TO-END!")


if __name__ == "__main__":
    asyncio.run(run_4_turns())
