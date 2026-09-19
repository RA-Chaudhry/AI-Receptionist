import sys, os
sys.path.insert(0, os.path.abspath("."))

import asyncio
import json
import edge_tts
import websockets
from app.utils.audio_buffer import mp3_to_pcm

async def simulate_browser_client():
    print("Generating speech audio...")
    comm = edge_tts.Communicate("Hello, who is the CEO of Logic Pulse?", "en-US-JennyNeural")
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            mp3_data += chunk.get("data", b"")
    pcm_data = mp3_to_pcm(mp3_data, target_rate=16000)
    print(f"Generated {len(pcm_data)} PCM bytes ({len(pcm_data)/32000:.2f}s)")
    
    uri = "ws://127.0.0.1:8000/api/v1/voice/stream?phone_number=test_caller"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("Connected! Streaming audio chunks...")
        chunk_size = 3200  # 100ms
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.05)
        
        print("Finished audio stream. Sending speech_ended event...")
        await ws.send(json.dumps({"type": "speech_ended"}))
        
        print("Listening for server responses...")
        received_transcript = False
        received_response_end = False
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=25.0)
                if isinstance(msg, str):
                    payload = json.loads(msg)
                    msg_type = payload.get("type")
                    msg_data = payload.get("data", "")
                    if msg_type == "transcript":
                        print(f"-> RECEIVED TRANSCRIPT: {msg_data.encode('utf-8')}")
                        received_transcript = True
                    elif msg_type == "response_start":
                        print("-> RECEIVED RESPONSE_START")
                    elif msg_type == "audio_chunk":
                        print(f"-> RECEIVED AUDIO_CHUNK (index={payload.get('index')})")
                    elif msg_type == "response_end":
                        print(f"-> RECEIVED RESPONSE_END: {msg_data.encode('utf-8')} (source={payload.get('source')}, provider={payload.get('provider')})")
                        received_response_end = True
                        break
                    elif msg_type == "error":
                        print(f"-> RECEIVED ERROR: '{msg_data}'")
                        break
                    else:
                        print(f"-> RECEIVED {msg_type}: {payload}")
                else:
                    print(f"-> RECEIVED BINARY: {len(msg)} bytes")
            except asyncio.TimeoutError:
                print("Timeout waiting for server response!")
                break
        
        print(f"Simulation completed! received_transcript={received_transcript}, received_response_end={received_response_end}")

if __name__ == "__main__":
    asyncio.run(simulate_browser_client())
