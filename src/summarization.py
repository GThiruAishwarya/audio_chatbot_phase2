# src/summarization.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re


def summarize_tfidf(text, num_sentences=8):
    """
    Summarize input text using TF-IDF sentence ranking.

    Args:
        text (str): The text to summarize.
        num_sentences (int): Number of sentences to include in summary.

    Returns:
        str: Extractive summary composed of most relevant sentences.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]
    if len(sentences) <= num_sentences:
        return text

    # Compute TF-IDF scores
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = tfidf.toarray().mean(axis=1)

    # Select top sentences
    top_idx = np.argsort(scores)[::-1][:num_sentences]
    summary = " ".join([sentences[i] for i in sorted(top_idx)])
    return summary
