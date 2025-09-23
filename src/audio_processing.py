# src/audio_processing.py
import os
import subprocess
import torch
import whisper
import pyttsx3

UPLOADS_DIR = "src/uploads"


def setup_ffmpeg():
    """Ensure ffmpeg is installed (used for audio conversion)."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError("FFmpeg is not installed or not in PATH.") from e


def save_audio_bytes(audio_bytes, filename="input.wav"):
    """Save uploaded/recorded audio to uploads folder."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path


def transcribe_with_whisper(audio_path):
    """Transcribe audio using Whisper (OpenAI)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def text_to_speech(text, filename="output.wav"):
    """Generate speech audio from text using pyttsx3 (offline)."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    path = os.path.join(UPLOADS_DIR, filename)

    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()

    return path
