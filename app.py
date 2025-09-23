# app.py
import os
import streamlit as st
import nltk
import numpy as np
import concurrent.futures
from nltk.data import find
from nltk.tokenize import sent_tokenize

from src.audio_processing import setup_ffmpeg, text_to_speech, save_audio_bytes, UPLOADS_DIR
from src.pipeline import process_audio_pipeline
from src.summarization import summarize_textrank
from src.semantic_qa import SemanticSearchQA, clean_text_for_training, split_sentences, deduplicate_semantic
from src.ui_components import render_audio_uploader, render_chat_input
from src.db_helpers import init_db, insert_sentence, get_all_sentences, clear_db


# ================= NLTK SETUP =================
try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# =============== CONFIGURATION ================
DB_PATH = "database.db"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
setup_ffmpeg()

st.set_page_config(page_title="WeiterEdge - Offline Audio Summarization & Q&A", layout="wide")
st.title("💬 WeiterEdge - Offline Audio Summarization & Semantic Q&A")

init_db(DB_PATH)


# ========== SESSION STATE VARIABLES ===========
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "semantic_qa" not in st.session_state:
    st.session_state.semantic_qa = SemanticSearchQA()
if "sentences" not in st.session_state:
    st.session_state.sentences = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_recording" not in st.session_state:
    st.session_state.voice_recording = None


# ================ HELPERS =====================
def split_questions(text):
    sentences = sent_tokenize(text)
    questions = [s.strip() for s in sentences if s.strip().endswith("?")]
    return questions if questions else sentences


def answer_question(semantic_qa, question):
    return question, semantic_qa.query(question, top_k=3)


def answer_questions_parallel(semantic_qa, questions):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(answer_question, semantic_qa, q) for q in questions]
        return [f.result() for f in futures]


def aggregate_answers(answers, max_length=500):
    """Combine multiple Q&A into a single coherent answer with semantic deduplication."""
    combined_text = " ".join([ans for _, ans in answers if ans and ans != "No relevant answer found."])
    if not combined_text:
        return "No relevant answer found."

    sentences = split_sentences(combined_text)
    if not sentences:
        return combined_text

    model = st.session_state.semantic_qa.model
    embeddings = model.encode(sentences, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dedup_sents = deduplicate_semantic(sentences, embeddings, threshold=0.85)
    final_answer = " ".join(dedup_sents)

    if len(final_answer) > max_length:
        final_answer = final_answer[:max_length] + "..."
    return final_answer


# ================== UI LAYOUT ==================
st.sidebar.header("⚙ Settings")

# ✅ Only Whisper available now
asr_engine = "whisper"
summarizer_engine = st.sidebar.selectbox("📝 Summarizer:", ["textrank", "bart"], index=0)

tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Training", "💬 Interaction"])


# ================== UPLOAD ==================
with tab1:
    st.subheader("Upload Audio File")
    uploaded_file, file_path = render_audio_uploader()


# ================== TRAINING ==================
with tab2:
    st.subheader("Transcript & Summary Generation")

    if uploaded_file and st.button("Process Audio (Transcription → Summarization)"):
        clear_db(DB_PATH)
        st.session_state.semantic_qa = SemanticSearchQA()
        st.session_state.sentences = []

        with st.spinner(f"Processing audio using {asr_engine.upper()} + {summarizer_engine.upper()}..."):
            # ✅ cleaned: no "engine" arg
            results = process_audio_pipeline(file_path, summarizer=summarizer_engine)
            st.session_state.transcript = results["raw_transcript"]
            st.session_state.summary = results["summary"]

        st.markdown(f"### Transcript ({asr_engine.upper()})")
        st.text_area("Transcript", st.session_state.transcript, height=200)
        st.download_button("Download Transcript", st.session_state.transcript, "transcript.txt")

        st.markdown(f"### Summary ({summarizer_engine.upper()})")
        st.text_area("Summary", st.session_state.summary, height=150)
        st.download_button("Download Summary", st.session_state.summary, "summary.txt")

        # Train Q&A engine
        if st.session_state.transcript:
            cleaned_transcript = clean_text_for_training(st.session_state.transcript)
            sentences = [s.strip() for s in nltk.sent_tokenize(cleaned_transcript) if s.strip()]

            for s in sentences:
                insert_sentence(DB_PATH, s)

            st.session_state.sentences = get_all_sentences(DB_PATH)
            st.session_state.semantic_qa.build_index(st.session_state.sentences)
            st.success("✅ Semantic search Q&A engine trained automatically on the new transcript!")


if not st.session_state.sentences:
    st.session_state.sentences = get_all_sentences(DB_PATH)
    if st.session_state.sentences:
        st.session_state.semantic_qa.build_index(st.session_state.sentences)


# ================== INTERACTION ==================
with tab3:
    # ---- Chat Input Bar ----
    text_query, voice_bytes, submit_text, submit_voice, attach_file = render_chat_input()

    # ---- Handle Text ----
    if submit_text and text_query.strip():
        questions = split_questions(text_query)
        answers = answer_questions_parallel(st.session_state.semantic_qa, questions)
        aggregated_answer = aggregate_answers(answers)

        st.session_state.chat_history.append(("user", text_query))
        st.session_state.chat_history.append(("bot", aggregated_answer))

    # ---- Handle Voice ----
    if submit_voice and voice_bytes:
        wav_path = save_audio_bytes(voice_bytes)
        # ✅ cleaned: no "engine" arg
        question_results = process_audio_pipeline(wav_path, summarizer=summarizer_engine)
        question_text = question_results["raw_transcript"]

        questions = split_questions(question_text)
        answers = answer_questions_parallel(st.session_state.semantic_qa, questions)
        aggregated_answer = aggregate_answers(answers)

        st.session_state.chat_history.append(("user", question_text))
        st.session_state.chat_history.append(("bot", aggregated_answer))
        st.session_state.voice_recording = None

    # ---- Show Chat ----
    if st.session_state.chat_history:
        chat_css = """
        <style>
        .chat-container { 
            max-width: 900px; 
            margin: auto; 
            max-height: 500px; 
            overflow-y: auto; 
            padding: 10px;
        }
        .user-msg, .bot-msg {
            padding: 12px 18px;
            border-radius: 10px;
            margin: 8px 0;
            max-width: 95%;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.5;
            background-color: #000000;  /* black */
            color: #ffffff;             /* white */
            text-align: left;
        }
        </style>
        """
        st.markdown(chat_css, unsafe_allow_html=True)

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="user-msg"> {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg"> {msg}</div>', unsafe_allow_html=True)
                if msg and msg != "No relevant answer found.":
                    answer_audio_path = text_to_speech(msg)
                    with open(answer_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)
