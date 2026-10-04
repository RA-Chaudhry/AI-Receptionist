#!/usr/bin/env python3
"""
scripts/e2e_voice_test.py — Real-Audio End-to-End Voice Latency & VAD Test Harness

Streams real pre-recorded 16kHz mono WAV files from tests/audio_samples/ in real time (20ms frames)
to a running AI Receptionist server via WebSocket, verifying:
1. 3s background noise pre-check (verifies no false turn triggers).
2. End-to-end TTFA (Time-To-First-Audio) latency measured from last audio frame sent to first audio_chunk received.
3. p50 and p95 statistical percentiles across 5 iterations per sample.
4. Clean JSON and tabular console output.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import wave
import numpy as np
import websockets

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "audio_samples"))

SAMPLES = [
    "01_hello.wav",
    "02_office_timings.wav",
    "03_office_location.wav",
    "04_book_appointment.wav",
    "05_services.wav",
    "06_roman_urdu.wav",
]


def load_wav_pcm_frames(wav_path: str, frame_duration_ms: int = 20) -> list[bytes]:
    """Reads a 16kHz mono WAV file and splits it into 20ms (640-byte) PCM frames."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sr != 16000 or ch != 1 or sw != 2:
        raise ValueError(f"File {wav_path} must be 16kHz mono 16-bit PCM (got {sr}Hz, {ch}ch, {sw*8}bit)")

    frame_size = int(16000 * (frame_duration_ms / 1000.0) * 2)  # 640 bytes
    frames = [raw[i : i + frame_size] for i in range(0, len(raw), frame_size)]
    return [f for f in frames if len(f) == frame_size]


async def run_single_benchmark_turn(ws_url: str, noise_frames: list[bytes], sample_frames: list[bytes], sample_name: str) -> dict:
    """
    Connects to the voice WebSocket, streams 3s noise (asserting no false trigger),
    then streams sample frames in real-time, measuring time to first audio response.
    """
    async with websockets.connect(ws_url) as ws:
        # Step 1: Stream 3 seconds of background noise to check for false triggers
        false_trigger_detected = False
        noise_stream_task = None

        for frame in noise_frames:
            await ws.send(frame)
            await asyncio.sleep(0.020)
            try:
                # Check if server erroneously dispatched a response
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
                msg = json.loads(msg_raw)
                if msg.get("type") in ("response_start", "audio_chunk", "transcript"):
                    false_trigger_detected = True
                    break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass

        if false_trigger_detected:
            return {
                "sample": sample_name,
                "status": "FAILED_NOISE_CHECK",
                "false_trigger": True,
                "ttfa_ms": None,
                "transcript": "",
                "bot_reply": "",
            }

        # Step 2: Stream sample audio frames in real time (20ms intervals)
        t_start_utterance = time.perf_counter()
        for frame in sample_frames:
            await ws.send(frame)
            await asyncio.sleep(0.020)

        # 400ms trailing silence to allow natural endpointing
        silence_frame = b"\x00" * 640
        for _ in range(20):
            await ws.send(silence_frame)
            await asyncio.sleep(0.020)

        t_audio_end = time.perf_counter()

        # Step 3: Await response and measure TTFA (time from audio end to first audio_chunk)
        first_audio_time = None
        transcript_text = ""
        bot_reply_text = ""
        source_type = ""
        timeout_limit = 15.0  # max 15s wait
        deadline = time.perf_counter() + timeout_limit

        while time.perf_counter() < deadline:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg_raw, bytes):
                    continue
                msg = json.loads(msg_raw)
                mtype = msg.get("type")

                if mtype == "transcript":
                    transcript_text = msg.get("data", "")
                elif mtype == "audio_chunk":
                    if first_audio_time is None and not msg.get("is_filler"):
                        first_audio_time = time.perf_counter()
                elif mtype == "response_end":
                    bot_reply_text = msg.get("data", "")
                    source_type = msg.get("source", "unknown")
                    break
                elif mtype == "goodbye":
                    break
            except asyncio.TimeoutError:
                break
            except websockets.ConnectionClosed:
                break

        ttfa_ms = round((first_audio_time - t_audio_end) * 1000.0, 1) if first_audio_time else None

        # Cleanly send goodbye
        try:
            await ws.send(json.dumps({"type": "goodbye"}))
        except Exception:
            pass

        return {
            "sample": sample_name,
            "status": "SUCCESS" if first_audio_time else "NO_AUDIO_RECEIVED",
            "false_trigger": False,
            "ttfa_ms": ttfa_ms,
            "source": source_type,
            "transcript": transcript_text,
            "bot_reply": bot_reply_text,
        }


async def main():
    parser = argparse.ArgumentParser(description="AI Receptionist Real-Audio End-to-End Voice Benchmark")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark iterations per sample (default: 5)")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON benchmark results")
    args = parser.parse_args()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[ERROR] GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set GROQ_API_KEY before running e2e_voice_test.py:", file=sys.stderr)
        print("  set GROQ_API_KEY=gsk_your_groq_key (Windows PowerShell: $env:GROQ_API_KEY='...')", file=sys.stderr)
        sys.exit(1)

    ws_url = f"ws://{args.host}:{args.port}/api/v1/voice/stream?phone_number=%2B923001234567"
    print(f"Connecting to: {ws_url}")
    print(f"Audio Samples Directory: {AUDIO_DIR}")
    print(f"Runs per sample: {args.runs}\n")

    noise_wav = os.path.join(AUDIO_DIR, "00_background_noise.wav")
    if not os.path.exists(noise_wav):
        print(f"[ERROR] Noise sample missing: {noise_wav}", file=sys.stderr)
        sys.exit(1)
    noise_frames = load_wav_pcm_frames(noise_wav)

    all_results = {}
    summary_table = []

    for sample_file in SAMPLES:
        sample_path = os.path.join(AUDIO_DIR, sample_file)
        if not os.path.exists(sample_path):
            print(f"[WARNING] Sample file missing: {sample_file} (skipping)")
            continue

        sample_frames = load_wav_pcm_frames(sample_path)
        latencies = []
        statuses = []
        sample_records = []

        print(f"Evaluating {sample_file} ({args.runs} iterations)...")
        for i in range(args.runs):
            res = await run_single_benchmark_turn(ws_url, noise_frames, sample_frames, sample_file)
            sample_records.append(res)
            statuses.append(res["status"])
            if res["ttfa_ms"] is not None:
                latencies.append(res["ttfa_ms"])
                print(f"  Run {i+1}: TTFA = {res['ttfa_ms']}ms | Source = {res.get('source')} | Bot: {res.get('bot_reply')[:40]}...")
            else:
                print(f"  Run {i+1}: FAILED ({res['status']})")
            await asyncio.sleep(0.5)

        p50 = float(np.percentile(latencies, 50)) if latencies else None
        p95 = float(np.percentile(latencies, 95)) if latencies else None
        min_l = min(latencies) if latencies else None
        max_l = max(latencies) if latencies else None

        all_results[sample_file] = {
            "runs": sample_records,
            "success_rate": f"{len(latencies)}/{args.runs}",
            "p50_ms": p50,
            "p95_ms": p95,
            "min_ms": min_l,
            "max_ms": max_l,
        }

        summary_table.append({
            "sample": sample_file,
            "success": f"{len(latencies)}/{args.runs}",
            "p50_ms": f"{p50:.1f}" if p50 is not None else "N/A",
            "p95_ms": f"{p95:.1f}" if p95 is not None else "N/A",
            "min_ms": f"{min_l:.1f}" if min_l is not None else "N/A",
            "max_ms": f"{max_l:.1f}" if max_l is not None else "N/A",
        })

    # Print markdown table
    print("\n" + "=" * 78)
    print("AI RECEPTIONIST REAL-AUDIO VOICE BENCHMARK RESULTS")
    print("=" * 78)
    header = f"{'Audio Sample':<28} | {'Success':<8} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9}"
    print(header)
    print("-" * len(header))
    for row in summary_table:
        print(f"{row['sample']:<28} | {row['success']:<8} | {row['p50_ms']:<9} | {row['p95_ms']:<9} | {row['min_ms']:<9} | {row['max_ms']:<9}")
    print("=" * 78 + "\n")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved full JSON results to: {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
