"""
tests/test_vad_turn_detection.py — Phase 4 VAD & Turn Detection Tests

Validates:
1. Short natural pause tolerance (does NOT cut off speaking turn)
2. Long pause completion (grace period triggers turn_completed)
3. Continuous speech tracking without interruption
4. Short noise/click rejection (durations < min_speech_duration_ms discarded)
5. Short single-word utterances ("Yes", "Okay") capture and completion
6. Maximum turn duration cutoff safety
7. Multi-session isolation (no state leakage between calls)
8. Barge-in / interruption buffer reset
"""

from unittest.mock import MagicMock
import pytest

from app.utils.audio_buffer import TurnDetector, TurnState, VADProcessor


class MockVADProcessor:
    """
    Mock VAD processor allowing fine-grained control over which frames are speech.
    """
    def __init__(self, default_speech: bool = True) -> None:
        self.default_speech = default_speech

    def is_speech(self, pcm_bytes: bytes) -> bool:
        return self.default_speech


def _create_frame(sample_rate: int = 16000, duration_ms: int = 20) -> bytes:
    """Creates a dummy 20ms PCM audio frame (640 bytes)."""
    frame_bytes = int(sample_rate * (duration_ms / 1000.0) * 2)
    return b"\x01\x00" * (frame_bytes // 2)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Natural Pause & Silence Grace Period Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_short_pause_does_not_terminate_turn():
    """
    User speaks for 1000ms, pauses for 300ms (< 700ms grace period), and speaks for 1000ms.
    The turn detector must remain in SPEAKING state throughout the pause.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        silence_duration_ms=700,
        min_speech_duration_ms=200,
    )
    frame = _create_frame()

    # 1. 1000ms speech (50 frames)
    vad.default_speech = True
    for _ in range(50):
        res = detector.process_frame(frame)
    assert detector.state == TurnState.SPEAKING

    # 2. 300ms natural pause / breath (15 frames)
    vad.default_speech = False
    for _ in range(15):
        res = detector.process_frame(frame)
        assert not res.turn_completed
        assert detector.state == TurnState.SPEAKING

    # 3. 1000ms speech resumes (50 frames)
    vad.default_speech = True
    for _ in range(50):
        res = detector.process_frame(frame)
        assert detector.state == TurnState.SPEAKING
        assert not res.turn_completed


def test_long_pause_terminates_turn():
    """
    User speaks for 1000ms, followed by 800ms of silence (> 700ms grace period).
    The turn detector must trigger turn_completed = True and return the full utterance.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        silence_duration_ms=700,
        min_speech_duration_ms=200,
    )
    frame = _create_frame()

    # 1. 1000ms speech (50 frames)
    vad.default_speech = True
    for _ in range(50):
        detector.process_frame(frame)
    assert detector.state == TurnState.SPEAKING

    # 2. Silence exceeding 700ms (35 frames = 700ms)
    vad.default_speech = False
    completed_res = None
    for i in range(40):
        res = detector.process_frame(frame)
        if res.turn_completed:
            completed_res = res
            break

    assert completed_res is not None
    assert completed_res.turn_completed is True
    assert completed_res.audio_bytes is not None
    assert len(completed_res.audio_bytes) > 0
    assert detector.state == TurnState.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# 2. Noise & Short Click Filtering Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_short_click_noise_is_ignored():
    """
    A short noise click of 80ms (4 frames < 200ms min_speech_duration_ms) followed by silence
    must be ignored and NOT transition to SPEAKING state.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        silence_duration_ms=700,
        min_speech_duration_ms=200,
    )
    frame = _create_frame()

    # 4 frames of noise (80ms)
    vad.default_speech = True
    for _ in range(4):
        res = detector.process_frame(frame)
        assert res.turn_started is False
        assert detector.state == TurnState.IDLE

    # Followed by silence (10 frames)
    vad.default_speech = False
    for _ in range(10):
        res = detector.process_frame(frame)
        assert res.turn_started is False
        assert detector.state == TurnState.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# 3. Short Utterance & Continuous Speech Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_short_utterance_captures_and_completes():
    """
    Short utterance ("Yes" / "Okay" ~300ms = 15 frames > 200ms min_speech) followed by silence.
    Detector must start turn, capture audio, and complete cleanly upon silence.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        silence_duration_ms=700,
        min_speech_duration_ms=200,
    )
    frame = _create_frame()

    # 300ms utterance
    vad.default_speech = True
    started = False
    for _ in range(15):
        res = detector.process_frame(frame)
        if res.turn_started:
            started = True
    assert started is True
    assert detector.state == TurnState.SPEAKING

    # Followed by silence (40 frames = 800ms)
    vad.default_speech = False
    completed = False
    for _ in range(40):
        res = detector.process_frame(frame)
        if res.turn_completed:
            completed = True
            break
    assert completed is True
    assert detector.state == TurnState.IDLE


def test_continuous_speech_stays_speaking():
    """
    Continuous speech for 4000ms (200 frames).
    Detector must remain in SPEAKING state and never terminate prematurely.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        max_turn_duration_ms=15000,
    )
    frame = _create_frame()

    vad.default_speech = True
    for i in range(200):
        res = detector.process_frame(frame)
        assert not res.turn_completed
        if i >= 10:  # after min_speech_duration_ms (200ms = 10 frames)
            assert detector.state == TurnState.SPEAKING


def test_max_turn_duration_cutoff():
    """
    Continuous speech exceeding max_turn_duration_ms (e.g. 5000ms in test)
    must trigger safety turn_completed cutoff.
    """
    vad = MockVADProcessor()
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        vad_processor=vad,
        max_turn_duration_ms=1000,  # 1 second max for test
    )
    frame = _create_frame()

    vad.default_speech = True
    completed = False
    for _ in range(60):  # 1200ms of speech
        res = detector.process_frame(frame)
        if res.turn_completed:
            completed = True
            break

    assert completed is True
    assert detector.state == TurnState.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# 4. Session Isolation & Barge-in Reset Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_turn_detector_session_isolation():
    """
    Verify two concurrent sessions have isolated VAD states without memory sharing.
    """
    vad_a = MockVADProcessor(default_speech=True)
    vad_b = MockVADProcessor(default_speech=False)

    detector_a = TurnDetector(vad_processor=vad_a, min_speech_duration_ms=100)
    detector_b = TurnDetector(vad_processor=vad_b, min_speech_duration_ms=100)
    frame = _create_frame()

    # Session A receives speech, Session B receives silence
    for _ in range(10):
        detector_a.process_frame(frame)
        detector_b.process_frame(frame)

    assert detector_a.state == TurnState.SPEAKING
    assert detector_b.state == TurnState.IDLE
    assert len(detector_b._active_utterance) == 0


def test_turn_detector_barge_in_reset():
    """
    Verify reset() drops active utterance and resets state to IDLE on barge-in.
    """
    vad = MockVADProcessor(default_speech=True)
    detector = TurnDetector(vad_processor=vad, min_speech_duration_ms=100)
    frame = _create_frame()

    # Enter SPEAKING state
    for _ in range(10):
        detector.process_frame(frame)
    assert detector.state == TurnState.SPEAKING
    assert len(detector._active_utterance) > 0

    # Barge-in interruption occurs
    detector.reset()
    assert detector.state == TurnState.IDLE
    assert len(detector._active_utterance) == 0
    assert len(detector._pre_buffer) == 0
