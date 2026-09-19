import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath("."))

import asyncio
import json
import math
import struct
import websockets
import edge_tts
from app.utils.audio_buffer import mp3_to_pcm

async def main():
    print("1. Generating Edge-TTS audio...", flush=True)
    try:
        comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
        mp3_data = b""
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                mp3_data += chunk.get("data", b"")
        pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    except Exception as e:
        print(f"Edge-TTS failed: {e}. Generating synthetic sine audio instead...", flush=True)
        # 16kHz, 1 sec sine tone
        pcm_data = bytearray()
        for i in range(16000 * 2):
            sample = int(10000 * math.sin(2 * math.pi * 440 * (i / 16000.0)))
            pcm_data.extend(struct.pack("<h", sample))
        pcm_data = bytes(pcm_data)

    print(f"2. Audio ready: {len(pcm_data)} bytes ({len(pcm_data)/32000:.2f}s)", flush=True)
    
    uri = "ws://127.0.0.1:8000/api/v1/voice/stream?phone_number=test_user"
    print(f"3. Connecting to {uri}...", flush=True)
    async with websockets.connect(uri) as ws:
        print("4. Connected! Sending audio in 100ms frames...", flush=True)
        chunk_size = 3200
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.04)
        
        print("5. Sending speech_ended event...", flush=True)
        await ws.send(json.dumps({"type": "speech_ended"}))

        print("6. Waiting for server events...", flush=True)
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                if isinstance(msg, str):
                    payload = json.loads(msg)
                    print(f"   [RECV JSON] type={payload.get('type')}, data={payload.get('data', '')[:60]}", flush=True)
                    if payload.get("type") in ("response_end", "goodbye", "timeout", "error"):
                        break
                else:
                    print(f"   [RECV BINARY] {len(msg)} bytes", flush=True)
            except asyncio.TimeoutError:
                print("   [TIMEOUT] Server took > 15s", flush=True)
                break

    print("7. Done!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
