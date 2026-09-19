import audioop
import asyncio
import base64
import contextlib
import io
import json
import queue
import sys
import threading

import pyaudio
import pygame
import websockets

WS_URL = "ws://localhost:8000/api/v1/voice/stream"
PHONE_NUMBER = "03001234567"
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 800
SEND_INTERVAL_SEC = 0.5


def mic_recording_thread(audio_queue: queue.Queue, stop_event: threading.Event) -> None:
    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        while not stop_event.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_queue.put(data)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


async def main() -> None:
    pygame.mixer.init(frequency=SAMPLE_RATE)
    audio_queue: queue.Queue[bytes] = queue.Queue()
    stop_event = threading.Event()

    print("=" * 60)
    print("AI Voice Client")
    print("=" * 60)

    mic_thread = threading.Thread(
        target=mic_recording_thread,
        args=(audio_queue, stop_event),
        daemon=True,
    )
    mic_thread.start()

    ws_uri = f"{WS_URL}?phone_number={PHONE_NUMBER}"
    try:
        async with websockets.connect(ws_uri, max_size=None) as ws:
            print("Connected! Start speaking...")

            async def send_audio() -> None:
                speaking = False
                silence_start = None
                noise_floor = 150.0  # Initial noise floor estimate
                
                while not stop_event.is_set():
                    try:
                        chunk = audio_queue.get_nowait()
                        rms = audioop.rms(chunk, 2)  # width is 2 for 16-bit PCM
                        now = asyncio.get_running_loop().time()
                        
                        # Adapt noise floor towards current RMS when quiet or slowly over time
                        if rms < noise_floor:
                            noise_floor = noise_floor * 0.8 + rms * 0.2
                        else:
                            noise_floor = noise_floor * 0.999 + rms * 0.001
                        
                        # Dynamic threshold: noise floor + 150 offset, but at least 300
                        threshold = max(300.0, noise_floor + 150.0)
                        
                        if rms > threshold:
                            if not speaking:
                                speaking = True
                                silence_start = None
                                # Interrupt AI voice if it was playing
                                pygame.mixer.stop()
                                await ws.send(json.dumps({"type": "interrupt"}))
                            else:
                                silence_start = None
                            await ws.send(chunk)
                        else:
                            if speaking:
                                await ws.send(chunk)
                                if silence_start is None:
                                    silence_start = now
                                elif now - silence_start > 1.5:
                                    speaking = False
                                    silence_start = None
                                    await ws.send(json.dumps({"type": "speech_ended"}))
                    except queue.Empty:
                        await asyncio.sleep(0.01)

            async def receive_responses() -> None:
                async for message in ws:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "transcript":
                        print(f"\n[YOU]: {msg.get('data', '')}")
                    elif msg_type == "response_start":
                        print(f"[BOT]: {msg.get('data', '')}")
                    elif msg_type == "audio_chunk":
                        audio_bytes = base64.b64decode(msg.get("data", ""))
                        if audio_bytes:
                            try:
                                sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
                                sound.play()
                            except Exception:
                                print("[WARN]: Could not play an audio chunk.")
                    elif msg_type == "response_end":
                        print("[BOT]: Response complete.")
                    elif msg_type == "goodbye":
                        print("[BOT]: Goodbye!")
                        stop_event.set()
                        break
                    elif msg_type == "error":
                        print(f"[ERROR]: {msg.get('data', '')}")
                    elif msg_type == "timeout":
                        print("[INFO]: Session timed out")
                        break

            sender_task = asyncio.create_task(send_audio())
            receiver_task = asyncio.create_task(receive_responses())

            try:
                await asyncio.gather(sender_task, receiver_task)
            finally:
                stop_event.set()
                sender_task.cancel()
                receiver_task.cancel()
                with contextlib.suppress(Exception):
                    await sender_task
                with contextlib.suppress(Exception):
                    await receiver_task
    finally:
        stop_event.set()
        pygame.mixer.quit()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
