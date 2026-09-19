import importlib
import platform
import subprocess
import sys
import tempfile
import time
import wave

PACKAGES = [
    ("faster-whisper", "faster_whisper"),
    ("webrtcvad", "webrtcvad"),
    ("edge-tts", "edge_tts"),
    ("websockets", "websockets"),
    ("pyaudio", "pyaudio"),
    ("numpy", "numpy"),
    ("pygame", "pygame"),
    ("soundfile", "soundfile"),
]


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    return (major, minor) >= (3, 9)


def install_package(pip_name: str, import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        print(f"{pip_name}: already installed")
        return True
    except Exception:
        pass

    print(f"{pip_name}: installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        importlib.import_module(import_name)
        print(f"{pip_name}: installed successfully")
        return True
    except Exception as exc:
        print(f"{pip_name}: failed ({exc})")
        return False


def download_whisper_model() -> bool:
    try:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
        elapsed = time.perf_counter() - started
        print(f"Whisper model loaded in {elapsed:.2f}s")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)

            segments, _info = model.transcribe(
                wav_path,
                beam_size=1,
                vad_filter=True,
                language="en",
            )
            _ = "".join(segment.text for segment in segments).strip()
            return True
        finally:
            try:
                import os

                os.unlink(wav_path)
            except Exception:
                pass
    except Exception as exc:
        print(f"Whisper model test failed: {exc}")
        return False


def test_microphone() -> bool:
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=800,
        )
        try:
            data = stream.read(800, exception_on_overflow=False)
            return len(data) > 0
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
    except Exception as exc:
        print(f"Microphone test failed: {exc}")
        return False


def test_edge_tts() -> bool:
    try:
        import asyncio
        import edge_tts

        async def _test() -> bool:
            communicate = edge_tts.Communicate("test", voice="en-US-JennyNeural")
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    return True
            return False

        return asyncio.run(_test())
    except Exception as exc:
        print(f"Edge TTS test failed: {exc}")
        return False


def print_summary(results: dict) -> None:
    print("\nSummary")
    print("-" * 40)
    for step, passed in results.items():
        mark = "✓" if passed else "✗"
        print(f"{step:<20} {mark}")

    if all(results.values()):
        print("\nRun python voice_client.py")
    else:
        print("\nOne or more checks failed.")
        print("Suggested fixes:")
        print("  python -m pip install -r requirements.txt")


def main() -> None:
    results = {}
    results["Python 3.9+"] = check_python_version()
    for pip_name, import_name in PACKAGES:
        results[pip_name] = install_package(pip_name, import_name)
    results["Whisper model"] = download_whisper_model()
    results["Microphone"] = test_microphone()
    results["Edge TTS"] = test_edge_tts()
    print_summary(results)


if __name__ == "__main__":
    main()
