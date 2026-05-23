"""
emotion_detector.py
───────────────────
Detects fine-grained emotions (joy, anger, sadness, fear, surprise, disgust,
neutral) from text using a pre-trained DistilRoBERTa emotion model from
Hugging Face.

Model: j-hartmann/emotion-english-distilroberta-base  (~330 MB, cached after
first download)
"""

from transformers import pipeline
import streamlit as st


# ── Cache the emotion pipeline so it loads only once ──────────────────────
@st.cache_resource
def _load_emotion_pipeline():
    """Load the emotion classification pipeline (cached across reruns)."""
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,          # Return scores for ALL emotion labels
        truncation=True,
    )


# ── Emoji map for display ────────────────────────────────────────────────
EMOTION_EMOJI = {
    "joy":      "😊",
    "anger":    "😠",
    "sadness":  "😢",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "neutral":  "😐",
}

# ── Color map for charts ─────────────────────────────────────────────────
EMOTION_COLOR = {
    "joy":      "#22c55e",
    "anger":    "#ef4444",
    "sadness":  "#3b82f6",
    "fear":     "#a855f7",
    "surprise": "#f59e0b",
    "disgust":  "#6b7280",
    "neutral":  "#94a3b8",
}


def detect_emotions(text: str) -> list[dict]:
    """
    Detect emotions in *text*.

    Returns a list of dicts sorted by descending score, e.g.:
        [{"label": "joy", "score": 0.92}, {"label": "anger", "score": 0.03}, …]
    """
    pipe = _load_emotion_pipeline()
    results = pipe(text)[0]  # top_k=None → list of dicts per input

    # Normalise label casing to lowercase for consistency
    for r in results:
        r["label"] = r["label"].lower()

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_top_emotion(results: list[dict]) -> tuple[str, float]:
    """Return (label, score) of the dominant emotion."""
    top = results[0]
    return top["label"], top["score"]
