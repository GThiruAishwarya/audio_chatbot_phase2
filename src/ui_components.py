# ui_components.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os

UPLOADS_DIR = "uploads"


def render_audio_uploader():
    uploaded_file = st.file_uploader(
        "Upload audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"]
    )
    if uploaded_file:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name.replace(" ", "_"))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(file_path)
        st.success(f"Uploaded: {uploaded_file.name}")
        return uploaded_file, file_path
    return None, None


def render_chat_input():
    """
    Chat input bar with:
    - Text field
    - Submit button
    - Mic (record audio)
    - Attach file
    """
    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])

    # Text input
    with col1:
        text_query = st.text_input(
            "Type your question here...",
            key="chat_input",
            label_visibility="collapsed"
        )

    # Submit button
    with col2:
        submit_text = st.button("📩", key="send_text_btn")

    # Mic (audio recorder widget, no button needed)
    with col3:
        recorded_audio = audio_recorder(
            text="",               # hide label text
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            icon_size="1.5x"
        )
        submit_voice = recorded_audio is not None

    # File attach
    with col4:
        attach_file = st.file_uploader(
            "📎", type=["txt", "pdf"], label_visibility="collapsed", key="attach_file"
        )

    return text_query, recorded_audio, submit_text, submit_voice, attach_file
