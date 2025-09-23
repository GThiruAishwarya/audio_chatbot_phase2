# pipeline.py
from src.audio_processing import transcribe_with_whisper
from src.summarization import summarize_textrank

# Try importing BART summarizer if available
try:
    from src.bart_summarizer import summarize_text as summarize_bart
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False


def process_audio_pipeline(file_path=None, summarizer="textrank"):
    """
    Full pipeline:
    1. Transcribe audio (Whisper)
    2. Summarize text (Textrank or BART if available)
    """
    transcript_text = ""
    if file_path:
        transcript_text = transcribe_with_whisper(file_path)

    summary = ""
    if transcript_text:
        if summarizer == "bart" and BART_AVAILABLE:
            summary = summarize_bart(transcript_text)
        else:
            summary = summarize_textrank(transcript_text)

    return {
        "raw_transcript": transcript_text,
        "summary": summary,
    }
