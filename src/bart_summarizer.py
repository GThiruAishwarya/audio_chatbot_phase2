# src/bart_summarizer.py
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model and tokenizer once
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")


def summarize_text(text: str) -> str:
    """
    Summarize cleaned transcript using BART.
    """
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
