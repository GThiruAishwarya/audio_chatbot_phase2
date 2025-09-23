# src/db_helpers.py
import sqlite3
import re


def init_db(db_path):
    """Create transcript_sentences table if not exists."""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS transcript_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT UNIQUE COLLATE NOCASE
            )"""
        )
        conn.commit()


def insert_sentence(db_path, sentence):
    """Insert a cleaned sentence into the DB if not a duplicate."""
    sentence = clean_sentence(sentence)
    if not sentence:
        return
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT OR IGNORE INTO transcript_sentences (sentence) VALUES (?)",
                (sentence,),
            )
            conn.commit()
    except Exception as e:
        print(f"⚠️ Skipped sentence due to error: {e}")


def insert_sentences_bulk(db_path, sentences):
    """Insert multiple sentences efficiently (ignores duplicates)."""
    cleaned = [clean_sentence(s) for s in sentences if s]
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.executemany(
            "INSERT OR IGNORE INTO transcript_sentences (sentence) VALUES (?)",
            [(s,) for s in cleaned if s],
        )
        conn.commit()


def get_all_sentences(db_path, with_ids=False):
    """Return all sentences stored in the DB."""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        if with_ids:
            c.execute("SELECT id, sentence FROM transcript_sentences")
            return c.fetchall()
        else:
            c.execute("SELECT sentence FROM transcript_sentences")
            return [row[0] for row in c.fetchall()]


def clear_db(db_path):
    """Delete all sentences from the DB."""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM transcript_sentences")
        conn.commit()


def clean_sentence(text):
    """
    Remove unwanted characters, extra spaces, and garbled text.
    """
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9,.!? ]+", "", text)
    return text if text else None
