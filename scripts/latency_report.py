#!/usr/bin/env python3
"""
scripts/latency_report.py — Telemetry Latency & Audio Duration Analysis Tool

Reads logs/telemetry.jsonl and computes/prints p50/p95 percentiles across latency stages
(endpointing, STT, LLM TTFT, First Clause, TTS TTFB, Network Send, Total from speech end)
and the distribution of turn audio duration (sec).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def percentile(data: List[float], p: float) -> float:
    """Computes the p-th percentile (0..100) of a list of floats using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def load_telemetry_records(log_path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = Path(log_path)
    if not path.exists():
        print(f"[!] Log file not found: {path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def analyze_latencies(records: List[Dict[str, Any]]) -> None:
    if not records:
        print("No telemetry records found to analyze.")
        return

    print("=" * 80)
    print(f"  AI RECEPTIONIST LATENCY & TELEMETRY REPORT ({len(records)} Turns)")
    print("=" * 80)

    # Categories
    llm_turns: List[Dict[str, Any]] = []
    router_turns: List[Dict[str, Any]] = []
    greeting_turns: List[Dict[str, Any]] = []

    for r in records:
        cost = r.get("cost", {})
        provider = cost.get("provider", "")
        turn_id = r.get("turn_id", "")
        channel = r.get("channel", "")

        if provider in ("local_router", "regex_rules") or "greeting" in turn_id:
            if "greeting" in turn_id:
                greeting_turns.append(r)
            else:
                router_turns.append(r)
        else:
            llm_turns.append(r)

    def extract_stage_metrics(dataset: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        stages: Dict[str, List[float]] = {
            "endpointing_ms": [],
            "stt_ms": [],
            "llm_ttft_ms": [],
            "first_clause_ms": [],
            "tts_ttfb_ms": [],
            "send_ms": [],
            "total_from_speech_end_ms": [],
            "total_turn_duration_ms": [],
            "audio_duration_sec": [],
        }

        for r in dataset:
            metrics = r.get("metrics_summary") or {}
            latency = r.get("latency") or {}

            # Audio duration
            aud_dur = r.get("audio_duration_sec") or 0.0
            if aud_dur > 0:
                stages["audio_duration_sec"].append(aud_dur)

            # Endpointing delay
            ep = metrics.get("endpointing_delay_ms") or latency.get("endpointing_delay_time_ms")
            if ep is not None:
                stages["endpointing_ms"].append(float(ep))

            # STT latency
            stt = metrics.get("stt_latency_ms")
            if stt is not None:
                stages["stt_ms"].append(float(stt))

            # LLM TTFT
            ttft = metrics.get("llm_ttft_ms")
            if ttft is not None:
                stages["llm_ttft_ms"].append(float(ttft))

            # First clause
            fc = metrics.get("first_clause_ms") or ttft
            if fc is not None:
                stages["first_clause_ms"].append(float(fc))

            # TTS TTFB
            tts = metrics.get("tts_ttfb_ms")
            if tts is not None:
                stages["tts_ttfb_ms"].append(float(tts))

            # Network send / buffering
            snd = metrics.get("send_ms") or metrics.get("network_buffering_ms")
            if snd is not None:
                stages["send_ms"].append(float(snd))

            # Total from speech end (TTFA)
            ttfa = metrics.get("total_from_speech_end_ms") or metrics.get("ttfa_ms")
            if ttfa is not None:
                stages["total_from_speech_end_ms"].append(float(ttfa))

            # Total turn duration
            tot = metrics.get("total_turn_duration_ms")
            if tot is not None:
                stages["total_turn_duration_ms"].append(float(tot))

        return stages

    def print_stage_table(title: str, dataset: List[Dict[str, Any]]) -> None:
        stages = extract_stage_metrics(dataset)
        print(f"\n--- {title} (Count: {len(dataset)}) ---")
        print(f"{'Stage / Metric':<32} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'Mean (ms)':>10} | {'Max (ms)':>10}")
        print("-" * 80)

        for key, name in [
            ("endpointing_ms", "Endpointing Delay (VAD)"),
            ("stt_ms", "STT Finalization"),
            ("llm_ttft_ms", "LLM Time-to-First-Token"),
            ("first_clause_ms", "First Clause Split"),
            ("tts_ttfb_ms", "TTS Time-to-First-Audio"),
            ("send_ms", "Websocket Send Duration"),
            ("total_from_speech_end_ms", "TOTAL from Speech End (TTFA)"),
            ("total_turn_duration_ms", "Total End-to-End Turn"),
        ]:
            vals = stages.get(key, [])
            if not vals:
                print(f"{name:<32} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
                continue
            p50 = percentile(vals, 50)
            p95 = percentile(vals, 95)
            mean = sum(vals) / len(vals)
            mx = max(vals)
            print(f"{name:<32} | {p50:>10.1f} | {p95:>10.1f} | {mean:>10.1f} | {mx:>10.1f}")

    if llm_turns:
        print_stage_table("LLM-Answered Turns (Target: p50 <= 2200ms, p95 <= 3000ms)", llm_turns)

    if router_turns or greeting_turns:
        canned = router_turns + greeting_turns
        print_stage_table("Router / Cached Turns (Target: <= 600ms)", canned)

    # Audio Duration Distribution
    all_stages = extract_stage_metrics(records)
    durations = all_stages.get("audio_duration_sec", [])
    if durations:
        print("\n--- Speech Audio Duration Distribution (sec) ---")
        p25 = percentile(durations, 25)
        p50 = percentile(durations, 50)
        p75 = percentile(durations, 75)
        p95 = percentile(durations, 95)
        print(f"Count: {len(durations)} | Min: {min(durations):.2f}s | p25: {p25:.2f}s | p50: {p50:.2f}s | p75: {p75:.2f}s | p95: {p95:.2f}s | Max: {max(durations):.2f}s")
    print("=" * 80)


def main() -> None:
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/telemetry.jsonl"
    records = load_telemetry_records(log_file)
    analyze_latencies(records)


if __name__ == "__main__":
    main()
