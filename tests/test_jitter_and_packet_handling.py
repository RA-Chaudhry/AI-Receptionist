import pytest
import math
import struct
from app.utils.audio_buffer import PacketSanitizer, JitterBuffer, TurnDetector, TurnState


def generate_pcm_sine(duration_ms: int = 20, sample_rate: int = 16000, freq: float = 440.0, amp: float = 20000.0) -> bytes:
    """Generates synthetic 16-bit mono PCM sine wave audio."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    samples = [
        int(amp * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_silence(duration_ms: int = 20, sample_rate: int = 16000) -> bytes:
    """Generates zeroed silence PCM frames."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    return b"\x00" * (num_samples * 2)


def test_packet_sanitizer_empty_and_invalid():
    """Verify PacketSanitizer safely handles None, empty, and non-byte inputs."""
    assert PacketSanitizer.sanitize_pcm_chunk(b"") == []
    assert PacketSanitizer.sanitize_pcm_chunk(None) == []  # type: ignore
    assert PacketSanitizer.sanitize_pcm_chunk(12345) == []  # type: ignore


def test_packet_sanitizer_odd_byte_alignment():
    """Verify PacketSanitizer drops orphan odd byte and zero-pads incomplete frames."""
    # 641 bytes (640 bytes frame + 1 orphan byte)
    raw = generate_pcm_sine(20) + b"\xFF"
    assert len(raw) == 641
    sanitized = PacketSanitizer.sanitize_pcm_chunk(raw, frame_bytes=640)
    assert len(sanitized) == 1
    assert len(sanitized[0]) == 640
    # Dropped the orphan byte, so the frame matches the original 640 bytes
    assert sanitized[0] == raw[:640]


def test_packet_sanitizer_oversized_chunk_safety():
    """Verify PacketSanitizer caps oversized packets to avoid memory explosion."""
    huge_chunk = b"\x01\x00" * 40000  # 80,000 bytes > 64KB
    sanitized = PacketSanitizer.sanitize_pcm_chunk(huge_chunk, frame_bytes=640, max_chunk_bytes=64 * 1024)
    total_bytes = sum(len(f) for f in sanitized)
    assert total_bytes <= 64 * 1024 + 640


def test_packet_sanitizer_burst_slicing():
    """Verify PacketSanitizer slices coalesced bursts into uniform 20ms frames."""
    # 5 frames of 20ms = 100ms = 3200 bytes
    burst = generate_pcm_sine(100)
    assert len(burst) == 3200
    frames = PacketSanitizer.sanitize_pcm_chunk(burst, frame_bytes=640)
    assert len(frames) == 5
    for f in frames:
        assert len(f) == 640


def test_jitter_buffer_in_order_delivery_and_latency():
    """Verify JitterBuffer holds target delay frames and pops in order."""
    jb = JitterBuffer(frame_bytes=640, target_delay_frames=2, max_buffer_frames=10)
    f0 = generate_pcm_sine(20, freq=300)
    f1 = generate_pcm_sine(20, freq=400)
    f2 = generate_pcm_sine(20, freq=500)

    # Push 1st frame (buffer depth 1 < target 2) -> pop is None
    jb.push(f0, sequence_num=0)
    assert jb.pop() is None

    # Push 2nd frame (buffer depth 2 == target 2) -> pop returns 1st frame
    jb.push(f1, sequence_num=1)
    popped0 = jb.pop()
    assert popped0 == f0

    # Push 3rd frame -> pop returns 2nd frame
    jb.push(f2, sequence_num=2)
    popped1 = jb.pop()
    assert popped1 == f1


def test_jitter_buffer_reorders_out_of_order_packets():
    """Verify JitterBuffer correctly reorders out-of-order sequence packets."""
    jb = JitterBuffer(frame_bytes=640, target_delay_frames=3, max_buffer_frames=10)
    f0 = b"\x01" * 640
    f1 = b"\x02" * 640
    f2 = b"\x03" * 640

    # Arrive out of order: seq 0, seq 2, seq 1
    jb.push(f0, sequence_num=0)
    jb.push(f2, sequence_num=2)
    jb.push(f1, sequence_num=1)

    assert jb.pop() == f0
    assert jb.pop() == f1
    assert jb.pop() == f2


def test_jitter_buffer_packet_loss_concealment():
    """Verify JitterBuffer generates silence frame for missing/dropped packet."""
    jb = JitterBuffer(frame_bytes=640, target_delay_frames=2, max_buffer_frames=10)
    f0 = b"\x11" * 640
    f2 = b"\x33" * 640
    # seq 1 was dropped / lost over network

    jb.push(f0, sequence_num=0)
    jb.push(f2, sequence_num=2)

    # Frame 0 returned
    assert jb.pop() == f0
    # Frame 1 missing -> PLC generates 640 bytes of zeroed silence
    plc_frame = jb.pop()
    assert plc_frame == b"\x00" * 640
    # Frame 2 returned next
    assert jb.pop() == f2


def test_jitter_buffer_overflow_protection():
    """Verify JitterBuffer bounds memory when receiving excess unpopped frames."""
    jb = JitterBuffer(frame_bytes=640, target_delay_frames=2, max_buffer_frames=4)
    for i in range(10):
        jb.push(b"\xaa" * 640, sequence_num=i)
    assert len(jb._buffer) == 4


def test_turn_detector_handles_packet_bursts_and_jitter():
    """Verify TurnDetector accurately detects speech turns from coalesced network bursts."""
    td = TurnDetector(silence_duration_ms=700, min_speech_duration_ms=200)

    # 1. Simulate burst of 300ms speech arriving in 1 single WebSocket packet
    speech_burst = generate_pcm_sine(300)
    result = td.process_frame(speech_burst)
    assert result.is_speech is True
    assert result.turn_started is True
    assert result.state == TurnState.SPEAKING

    # 2. Simulate 800ms silence arriving in two 400ms burst chunks
    silence_burst_1 = generate_silence(400)
    res1 = td.process_frame(silence_burst_1)
    assert res1.turn_completed is False

    silence_burst_2 = generate_silence(400)
    res2 = td.process_frame(silence_burst_2)
    assert res2.turn_completed is True
    assert res2.audio_bytes is not None
    assert len(res2.audio_bytes) >= int(16000 * 0.3 * 2)  # At least 300ms captured


def test_turn_detector_malformed_chunk_resilience():
    """Verify TurnDetector handles corrupt, odd-length, and empty chunks without crashing."""
    td = TurnDetector()
    res1 = td.process_frame(b"")
    assert res1.turn_started is False
    assert res1.is_speech is False

    odd_corrupt = b"\x01\x02\x03"
    res2 = td.process_frame(odd_corrupt)
    assert res2.turn_started is False
