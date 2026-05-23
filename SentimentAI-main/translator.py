"""
translator.py
─────────────
Auto-detects the language of a review and translates it to English before
sentiment analysis.

Dependencies:
    - langdetect  — fast language identification
    - deep-translator  — Google Translate wrapper (free, no API key)
"""

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Make langdetect deterministic
DetectorFactory.seed = 0

# Map ISO 639-1 codes to readable names for display
LANG_NAMES = {
    "en": "English",   "es": "Spanish",    "fr": "French",
    "de": "German",    "it": "Italian",    "pt": "Portuguese",
    "nl": "Dutch",     "ru": "Russian",    "ja": "Japanese",
    "ko": "Korean",    "zh-cn": "Chinese", "zh-tw": "Chinese (Traditional)",
    "ar": "Arabic",    "hi": "Hindi",      "bn": "Bengali",
    "tr": "Turkish",   "pl": "Polish",     "sv": "Swedish",
    "da": "Danish",    "no": "Norwegian",  "fi": "Finnish",
    "th": "Thai",      "vi": "Vietnamese", "id": "Indonesian",
    "ms": "Malay",     "tl": "Filipino",   "uk": "Ukrainian",
    "cs": "Czech",     "ro": "Romanian",   "hu": "Hungarian",
    "el": "Greek",     "he": "Hebrew",     "te": "Telugu",
    "ta": "Tamil",     "mr": "Marathi",    "gu": "Gujarati",
    "kn": "Kannada",   "ml": "Malayalam",  "pa": "Punjabi",
    "ur": "Urdu",
}


def detect_and_translate(text: str) -> tuple[str, str, bool]:
    """
    Detect the language of *text* and translate to English if necessary.

    Returns
    -------
    translated_text : str
        The English version of the text (unchanged if already English).
    source_language : str
        Human-readable name of the detected language.
    was_translated : bool
        True if translation was performed.
    """
    try:
        lang_code = detect(text)
    except Exception:
        # If detection fails, assume English
        return text, "English", False

    lang_name = LANG_NAMES.get(lang_code, lang_code.upper())

    if lang_code == "en":
        return text, "English", False

    try:
        translated = GoogleTranslator(source=lang_code, target="en").translate(text)
        return translated, lang_name, True
    except Exception:
        # If translation fails, return original text
        return text, lang_name, False
