from __future__ import annotations

import io
import logging
import wave

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None

try:
    import webrtcvad
except Exception:  # pragma: no cover - optional dependency
    webrtcvad = None

logger = logging.getLogger(__name__)


from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

try:
    from app.core.config import settings
except Exception:
    settings = None


class TurnState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


@dataclass
class TurnDetectorResult:
    is_speech: bool
    turn_started: bool
    turn_completed: bool
    audio_bytes: bytes | None = None
    state: TurnState = TurnState.IDLE


class CircularAudioBuffer:
    def __init__(self, max_seconds: int = 2, sample_rate: int = 16000):
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self._max_bytes = int(max_seconds * sample_rate * 2)
        self._buffer = bytearray()

    def add_chunk(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return

        self._buffer.extend(pcm_bytes)
        overflow = len(self._buffer) - self._max_bytes
        if overflow > 0:
            del self._buffer[:overflow]

    def get_last_n_seconds(self, n: float) -> bytes:
        if n <= 0 or not self._buffer:
            return b""

        byte_count = int(min(len(self._buffer), n * self.sample_rate * 2))
        return bytes(self._buffer[-byte_count:])

    def clear(self) -> None:
        self._buffer.clear()


class VADProcessor:
    def __init__(self, aggressiveness: int | None = None, sample_rate: int = 16000):
        self.aggressiveness = aggressiveness if aggressiveness is not None else (
            getattr(settings, "vad_aggressiveness", 1) if settings else 1
        )
        self.sample_rate = sample_rate
        self.vad = None

        if webrtcvad is None:
            logger.warning("webrtcvad is unavailable; falling back to fail-open VAD.")
            return

        try:
            self.vad = webrtcvad.Vad(self.aggressiveness)
        except Exception:
            logger.warning("Failed to initialize webrtcvad; falling back to fail-open VAD.", exc_info=True)
            self.vad = None

    def is_speech(self, pcm_bytes: bytes) -> bool:
        if not pcm_bytes:
            return False
        if self.has_silence(pcm_bytes, threshold_rms=30):
            return False
        if self.vad is None:
            return not self.has_silence(pcm_bytes, threshold_rms=60)

        # Accept 20ms (640 bytes) or 30ms (960 bytes) frames at 16kHz
        frame_size = int(self.sample_rate * 0.02) * 2  # 640 bytes for 20ms
        frames = [pcm_bytes[i : i + frame_size] for i in range(0, len(pcm_bytes), frame_size)]
        frames = [frame for frame in frames if len(frame) == frame_size]
        if not frames:
            # Try 30ms fallback (960 bytes)
            frame_size = int(self.sample_rate * 0.03) * 2
            frames = [pcm_bytes[i : i + frame_size] for i in range(0, len(pcm_bytes), frame_size)]
            frames = [frame for frame in frames if len(frame) == frame_size]
            if not frames:
                return False

        speech_frames = 0
        for frame in frames:
            try:
                is_vad_speech = self.vad.is_speech(frame, self.sample_rate)
                has_solid_energy = not self.has_silence(frame, threshold_rms=50)
                if is_vad_speech or has_solid_energy:
                    speech_frames += 1
            except Exception:
                logger.debug("VAD frame processing failed", exc_info=True)

        threshold = getattr(settings, "vad_speech_threshold", 0.3) if settings else 0.3
        return (speech_frames / len(frames)) >= threshold

    def has_silence(self, pcm_bytes: bytes, threshold_rms: float = 60) -> bool:
        if not pcm_bytes:
            return True

        if np is not None:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return True
            rms = float(np.sqrt(np.mean(np.square(samples))))
            return rms < threshold_rms

        total = 0.0
        sample_count = 0
        for index in range(0, len(pcm_bytes) - 1, 2):
            sample = int.from_bytes(pcm_bytes[index : index + 2], byteorder="little", signed=True)
            total += float(sample * sample)
            sample_count += 1
        if sample_count == 0:
            return True
        rms = (total / sample_count) ** 0.5
        return rms < threshold_rms


class PacketSanitizer:
    """
    Robust sanitizer for incoming raw PCM audio packets.
    Handles malformed frames, odd byte lengths, bursts, and extreme packet sizes.
    """
    @staticmethod
    def sanitize_pcm_chunk(
        raw_bytes: bytes,
        frame_bytes: int = 640,
        max_chunk_bytes: int = 64 * 1024,
    ) -> list[bytes]:
        """
        Takes raw incoming bytes and sanitizes them:
        1. Filters empty or non-byte data.
        2. Limits extreme/oversized chunks to avoid memory spikes.
        3. Enforces 16-bit PCM byte alignment (removes dangling orphan byte if odd length).
        4. Slices coalesced bursts into standard frame_bytes chunks (pads last frame with silence if incomplete).
        """
        if not raw_bytes or not isinstance(raw_bytes, (bytes, bytearray)):
            return []

        # Prevent huge memory bursts / DoS from rogue packets
        if len(raw_bytes) > max_chunk_bytes:
            raw_bytes = raw_bytes[:max_chunk_bytes]

        # Enforce 16-bit alignment (sample is 2 bytes)
        if len(raw_bytes) % 2 != 0:
            raw_bytes = raw_bytes[:-1]

        if not raw_bytes:
            return []

        # Slice into standardized frame chunks
        frames = []
        for i in range(0, len(raw_bytes), frame_bytes):
            frame = bytes(raw_bytes[i : i + frame_bytes])
            if len(frame) < frame_bytes:
                # Zero-pad incomplete trailing frame for VAD compatibility
                frame = frame + b"\x00" * (frame_bytes - len(frame))
            frames.append(frame)

        return frames


class JitterBuffer:
    """
    Adaptive / Fixed-depth Jitter Buffer for raw audio streaming.
    Smooths packet arrival jitter, reorders sequenced frames, and provides
    Packet Loss Concealment (PLC) / zero-padding for dropped frames.
    """
    def __init__(
        self,
        frame_bytes: int = 640,
        target_delay_frames: int = 2,  # 2 frames = 40ms buffer
        max_buffer_frames: int = 10,   # 200ms max buffer
    ):
        self.frame_bytes = frame_bytes
        self.target_delay_frames = target_delay_frames
        self.max_buffer_frames = max_buffer_frames
        self._buffer: list[tuple[int, bytes]] = []  # (sequence_num, frame_bytes)
        self._next_expected_seq = 0
        self._initialized = False
        self._playing = False

    def push(self, frame_bytes: bytes, sequence_num: int | None = None) -> None:
        """Adds a frame with an optional sequence number."""
        if not frame_bytes:
            return

        # Ensure frame is properly sized
        if len(frame_bytes) < self.frame_bytes:
            frame_bytes = frame_bytes + b"\x00" * (self.frame_bytes - len(frame_bytes))
        elif len(frame_bytes) > self.frame_bytes:
            frame_bytes = frame_bytes[:self.frame_bytes]

        if sequence_num is None:
            # Synthetic sequential indexing for raw arrival stream
            if not self._initialized:
                self._next_expected_seq = 0
                self._initialized = True
                seq = 0
            else:
                seq = (self._buffer[-1][0] + 1) if self._buffer else self._next_expected_seq
        else:
            if not self._initialized:
                self._next_expected_seq = sequence_num
                self._initialized = True
            seq = sequence_num

        # Insert sorted by sequence number
        self._buffer.append((seq, frame_bytes))
        self._buffer.sort(key=lambda x: x[0])

        if len(self._buffer) >= self.target_delay_frames:
            self._playing = True

        # Drop oldest if buffer overflows beyond max
        if len(self._buffer) > self.max_buffer_frames:
            overflow = len(self._buffer) - self.max_buffer_frames
            del self._buffer[:overflow]

    def pop(self) -> bytes | None:
        """
        Pops the next in-order frame.
        If buffer has reached target delay:
        - If next expected packet is present, return it.
        - If packet is missing (dropped), perform Packet Loss Concealment (silence/zero frame) and increment sequence.
        """
        if not self._playing:
            if len(self._buffer) < self.target_delay_frames:
                return None
            self._playing = True

        if not self._buffer:
            self._playing = False
            return None

        current_seq, frame = self._buffer[0]
        if current_seq <= self._next_expected_seq:
            self._buffer.pop(0)
            self._next_expected_seq = current_seq + 1
            return frame
        else:
            # Packet loss detected (gap in sequence) -> PLC with comfort silence frame
            self._next_expected_seq += 1
            return b"\x00" * self.frame_bytes

    def flush(self) -> list[bytes]:
        """Flushes and returns all buffered frames in sequence."""
        frames = [f for _, f in self._buffer]
        self._buffer.clear()
        self._playing = False
        return frames

    def clear(self) -> None:
        """Clears all buffered frames and resets state."""
        self._buffer.clear()
        self._initialized = False
        self._next_expected_seq = 0
        self._playing = False


class TurnDetector:
    """
    Session-isolated, stateful Voice Activity and Turn Detector.
    Distinguishes between active speech, natural pauses, end of turn, noise clicks, and barge-in.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        vad_processor: VADProcessor | None = None,
        speech_threshold: float | None = None,
        silence_duration_ms: int | None = None,
        min_speech_duration_ms: int | None = None,
        max_turn_duration_ms: int | None = None,
        pre_buffer_ms: int | None = None,
        post_buffer_ms: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * 2)

        self.speech_threshold = speech_threshold if speech_threshold is not None else (
            getattr(settings, "vad_speech_threshold", 0.5) if settings else 0.5
        )
        self.silence_duration_ms = silence_duration_ms if silence_duration_ms is not None else (
            getattr(settings, "vad_silence_duration_ms", 700) if settings else 700
        )
        self.min_speech_duration_ms = min_speech_duration_ms if min_speech_duration_ms is not None else (
            getattr(settings, "vad_min_speech_duration_ms", 60) if settings else 60
        )
        self.max_turn_duration_ms = max_turn_duration_ms if max_turn_duration_ms is not None else (
            getattr(settings, "vad_max_turn_duration_ms", 15000) if settings else 15000
        )
        self.pre_buffer_ms = pre_buffer_ms if pre_buffer_ms is not None else (
            getattr(settings, "vad_pre_buffer_ms", 300) if settings else 300
        )
        self.post_buffer_ms = post_buffer_ms if post_buffer_ms is not None else (
            getattr(settings, "vad_post_buffer_ms", 200) if settings else 200
        )

        self.vad_processor = vad_processor or VADProcessor(
            aggressiveness=getattr(settings, "vad_aggressiveness", 1) if settings else 1,
            sample_rate=sample_rate,
        )

        # Strictly bounded circular pre-buffer to prevent memory leaks during idle
        pre_frames_count = max(1, int(self.pre_buffer_ms / self.frame_duration_ms))
        self._pre_buffer: deque[bytes] = deque(maxlen=pre_frames_count)

        self.state = TurnState.IDLE
        self._active_utterance = bytearray()
        self._speech_frames_count = 0
        self._silence_frames_count = 0
        self._pending_speech_frames: list[bytes] = []
        self._pending_silence_count = 0

    def reset(self) -> None:
        """Resets all internal buffers and state (e.g. on barge-in or session start)."""
        self.state = TurnState.IDLE
        self._pre_buffer.clear()
        self._active_utterance.clear()
        self._speech_frames_count = 0
        self._silence_frames_count = 0
        self._pending_speech_frames.clear()
        self._pending_silence_count = 0

    def process_frame(self, pcm_data: bytes) -> TurnDetectorResult:
        """
        Processes an incoming audio chunk (or sequence of frames) and evaluates turn state.
        Returns TurnDetectorResult with status flags and audio bytes if turn is complete.
        """
        if not pcm_data:
            return TurnDetectorResult(
                is_speech=False,
                turn_started=False,
                turn_completed=False,
                state=self.state,
            )

        frames = PacketSanitizer.sanitize_pcm_chunk(pcm_data, self.frame_bytes)
        if not frames:
            return TurnDetectorResult(
                is_speech=False,
                turn_started=False,
                turn_completed=False,
                state=self.state,
            )

        overall_is_speech = False
        turn_started_flag = False
        turn_completed_flag = False
        completed_audio: bytes | None = None

        min_speech_frames = max(1, int(self.min_speech_duration_ms / self.frame_duration_ms))
        max_silence_frames = max(1, int(self.silence_duration_ms / self.frame_duration_ms))
        max_turn_bytes = int(self.sample_rate * (self.max_turn_duration_ms / 1000.0) * 2)

        for frame in frames:

            is_frame_speech = self.vad_processor.is_speech(frame)
            if is_frame_speech:
                overall_is_speech = True

            if self.state == TurnState.IDLE:
                if is_frame_speech:
                    self._pending_speech_frames.append(frame)
                    self._pending_silence_count = 0
                    # Check if accumulated speech exceeds minimum threshold (noise/click filter)
                    if len(self._pending_speech_frames) >= min_speech_frames:
                        self.state = TurnState.SPEAKING
                        turn_started_flag = True
                        self._speech_frames_count = len(self._pending_speech_frames)
                        self._silence_frames_count = 0

                        # Prepend circular pre-buffer frames to prevent clipping initial consonants
                        for pre_frame in self._pre_buffer:
                            self._active_utterance.extend(pre_frame)
                        self._pre_buffer.clear()

                        # Append pending validated speech frames
                        for pending_frame in self._pending_speech_frames:
                            self._active_utterance.extend(pending_frame)
                        self._pending_speech_frames.clear()
                else:
                    # Allow 2 frames of tolerance before discarding pending speech frames
                    if self._pending_speech_frames:
                        self._pending_silence_count += 1
                        if self._pending_silence_count > 2:
                            self._pending_speech_frames.clear()
                            self._pending_silence_count = 0
                    self._pre_buffer.append(frame)

            elif self.state == TurnState.SPEAKING:
                self._active_utterance.extend(frame)

                if is_frame_speech:
                    self._speech_frames_count += 1
                    self._silence_frames_count = 0
                else:
                    self._silence_frames_count += 1

                    # Check if silence exceeds grace period (end of turn)
                    if self._silence_frames_count >= max_silence_frames:
                        turn_completed_flag = True
                        completed_audio = bytes(self._active_utterance)
                        self.reset()
                        break

                # Max turn duration guard
                if len(self._active_utterance) >= max_turn_bytes:
                    turn_completed_flag = True
                    completed_audio = bytes(self._active_utterance)
                    self.reset()
                    break

        return TurnDetectorResult(
            is_speech=overall_is_speech or (self.state == TurnState.SPEAKING),
            turn_started=turn_started_flag,
            turn_completed=turn_completed_flag,
            audio_bytes=completed_audio,
            state=self.state,
        )


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    return pcm_bytes, sample_rate


def mp3_to_pcm(mp3_bytes: bytes, target_rate: int = 16000) -> bytes:
    if sf is None or np is None:
        return mp3_bytes

    try:
        with io.BytesIO(mp3_bytes) as buffer:
            audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)

        if audio is None:
            return b""

        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)

        if sample_rate != target_rate and len(audio) > 1:
            duration = len(audio) / sample_rate
            target_length = max(1, int(duration * target_rate))
            source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
            audio = np.interp(target_positions, source_positions, audio)

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        return pcm.tobytes()
    except Exception:
        logger.exception("MP3 to PCM conversion failed; returning original bytes")
        return mp3_bytes
