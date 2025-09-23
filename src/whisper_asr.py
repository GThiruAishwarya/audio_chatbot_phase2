import whisper
#whisper_asr.py code
def transcribe_with_whisper(audio_file: str, model_name: str = "base") -> str:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_file: Path to the audio file (wav, mp3, m4a, etc.)
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")

    Returns:
        str: Transcript text
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return result["text"]
