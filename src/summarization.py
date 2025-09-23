'''
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def summarize_text(text, num_sentences=8):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]
    if len(sentences) <= num_sentences:
        return text
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = tfidf.toarray().mean(axis=1)
    top_idx = np.argsort(scores)[::-1][:num_sentences]
    summary = "\n".join([sentences[i] for i in sorted(top_idx)])
    return summary'''
import spacy#summarization.py code
import pytextrank

# Load spaCy model and add PyTextRank pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def summarize_textrank(text, limit_sentences=5):
    """
    Summarizes input text using TextRank algorithm.

    Args:
        text (str): Text to summarize.
        limit_sentences (int): Number of sentences to include in summary.

    Returns:
        str: Extractive summary composed of most important sentences.
    """
    doc = nlp(text)
    summary_sentences = [sent.text for sent in doc._.textrank.summary(limit_sentences=limit_sentences)]
    return " ".join(summary_sentences)
