import numpy as np
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
import re
import nltk
import faiss
#semantic_qa.py code
# ===============================
# Utility Functions
# ===============================

def split_sentences(text: str):
    """Split text into individual sentences using NLTK."""
    return [s.strip() for s in nltk.sent_tokenize(text.strip()) if s.strip()]


def sliding_window_chunks(sentences, window_size=4, stride=2):
    """Create overlapping chunks of text using a sliding window."""
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = sentences[i:i + window_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + window_size >= len(sentences):
            break
    return chunks


def clean_text_for_training(text: str):
    """Clean text by removing extra spaces and unwanted characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    words = text.split()
    cleaned_words, prev_word = [], None
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
        prev_word = word
    return " ".join(cleaned_words).strip()


def deduplicate_semantic(sentences, embeddings, threshold=0.85):
    """Remove semantically similar sentences using cosine similarity."""
    selected, selected_embeddings = [], []
    for sent, emb in zip(sentences, embeddings):
        if not selected_embeddings:
            selected.append(sent)
            selected_embeddings.append(emb)
            continue
        sims = np.dot(selected_embeddings, emb)  # cosine similarity
        if max(sims) < threshold:
            selected.append(sent)
            selected_embeddings.append(emb)
    return selected


# ===============================
# Main Semantic Search QA Class
# ===============================

class SemanticSearchQA:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the semantic search system with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None

    def build_index(self, sentences, window_size=4, stride=2):
        """Build the FAISS index from transcript sentences."""
        cleaned_sents = [clean_text_for_training(s) for s in sentences if s.strip()]
        cleaned_sents = list(dict.fromkeys(cleaned_sents))  # remove exact duplicates

        self.chunks = sliding_window_chunks(cleaned_sents, window_size, stride)
        if not self.chunks:
            print("⚠️ No chunks generated from transcript.")
            return

        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"✅ Built FAISS index with {len(self.chunks)} chunks.")

    def query(self, question, top_k=7, semantic_weight=0.9, fuzzy_weight=0.1, score_threshold=0.3):
        """Query the index to find the most relevant answer."""

        # ✅ Fail-safe: check if index is built
        if self.index is None or not self.chunks:
            return "No knowledge base is loaded. Please upload a transcript and train first."

        # Encode and normalize question
        q_emb = self.model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # Search top-k
        D, I = self.index.search(q_emb, top_k)

        candidates = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            candidate_text = self.chunks[idx]

            fuzzy_score = fuzz.token_set_ratio(question.lower(), candidate_text.lower()) / 100.0
            combined_score = semantic_weight * dist + fuzzy_weight * fuzzy_score

            if combined_score >= score_threshold:
                candidates.append((combined_score, candidate_text))

        print(f"🔎 Query: {question}")
        print(f"   Retrieved {len(candidates)} candidate chunks.")

        # ✅ If no good candidates → return "No relevant answer found"
        if not candidates:
            return "No relevant answer found."

        # Extract sentences from candidate chunks
        all_sentences, all_embeddings = [], []
        for _, chunk in candidates:
            for s in split_sentences(chunk):
                fuzz_score = fuzz.token_set_ratio(question.lower(), s.lower()) / 100.0
                if fuzz_score >= 0.3:
                    all_sentences.append(s)
                    emb = self.model.encode([s], convert_to_numpy=True)
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    all_embeddings.append(emb[0])

        # Deduplicate and return
        if all_sentences:
            dedup_sents = deduplicate_semantic(all_sentences, all_embeddings, threshold=0.85)
            final_answer = " ".join(dedup_sents)
            return clean_text_for_training(final_answer)

        return "No relevant answer found."
