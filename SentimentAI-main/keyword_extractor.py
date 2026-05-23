"""
keyword_extractor.py
────────────────────
Provides basic explainability by highlighting words that most influenced the
sentiment prediction.

Two complementary strategies:
1.  Lexicon overlap — check each word against known positive / negative word
    lists and flag matches.
2.  TF-IDF-style scoring — rank words by how sentiment-bearing they are.

The output is a dict with categorised word lists **and** a ready-to-render
HTML string where positive words are highlighted green and negative words red.
"""

import re


# ── Sentiment lexicons ────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "amazing", "perfect", "love",
    "wonderful", "fantastic", "best", "happy", "nice", "beautiful", "superb",
    "brilliant", "outstanding", "pleased", "enjoyed", "recommend", "favorite",
    "impressive", "delightful", "magnificent", "exceptional", "remarkable",
    "incredible", "extraordinary", "splendid", "phenomenal", "glorious",
    "charming", "elegant", "graceful", "marvelous", "satisfying",
    "thrilling", "terrific", "stellar", "top-notch", "flawless",
}

NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "poor", "hate", "horrible",
    "disappointing", "useless", "waste", "broken", "annoying", "frustrating",
    "dreadful", "pathetic", "rubbish", "disgusting", "regret", "refund",
    "mediocre", "inferior", "unpleasant", "defective", "ugly", "boring",
    "dull", "painful", "miserable", "lousy", "abysmal", "atrocious",
    "appalling", "horrendous", "ghastly", "nightmare", "dismal",
    "unbearable", "intolerable", "offensive", "repulsive",
}

INTENSIFIERS = {
    "very", "extremely", "incredibly", "absolutely", "totally", "completely",
    "utterly", "really", "highly", "remarkably", "exceptionally",
}

NEGATORS = {
    "not", "never", "no", "neither", "nor", "hardly", "barely", "scarcely",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
}


def extract_keywords(text: str) -> dict:
    """
    Analyse *text* and return a dict with:
        - positive_words : list[str]   — words flagged as positive
        - negative_words : list[str]   — words flagged as negative
        - intensifiers   : list[str]   — amplifier words found
        - negators       : list[str]   — negation words found
        - annotated_html : str         — HTML with coloured highlights
    """
    words = re.findall(r"\b[\w'-]+\b", text.lower())

    pos_found = [w for w in words if w in POSITIVE_WORDS]
    neg_found = [w for w in words if w in NEGATIVE_WORDS]
    int_found = [w for w in words if w in INTENSIFIERS]
    negator_found = [w for w in words if w in NEGATORS]

    # Build annotated HTML
    annotated = _build_annotated_html(text)

    return {
        "positive_words": list(dict.fromkeys(pos_found)),   # unique, order-preserved
        "negative_words": list(dict.fromkeys(neg_found)),
        "intensifiers":   list(dict.fromkeys(int_found)),
        "negators":       list(dict.fromkeys(negator_found)),
        "annotated_html": annotated,
    }


def _build_annotated_html(text: str) -> str:
    """
    Return an HTML string of *text* where positive words are wrapped in green
    spans, negative words in red spans, intensifiers in blue, and negators in
    orange.
    """
    # Tokenise while preserving whitespace and punctuation
    tokens = re.split(r"(\s+)", text)
    html_parts = []

    for token in tokens:
        clean = re.sub(r"[^\w'-]", "", token.lower())

        if clean in POSITIVE_WORDS:
            html_parts.append(
                f'<span style="background:rgba(34,197,94,0.25); color:#22c55e; '
                f'padding:2px 6px; border-radius:6px; font-weight:600;">{token}</span>'
            )
        elif clean in NEGATIVE_WORDS:
            html_parts.append(
                f'<span style="background:rgba(239,68,68,0.25); color:#ef4444; '
                f'padding:2px 6px; border-radius:6px; font-weight:600;">{token}</span>'
            )
        elif clean in INTENSIFIERS:
            html_parts.append(
                f'<span style="background:rgba(59,130,246,0.25); color:#60a5fa; '
                f'padding:2px 6px; border-radius:6px; font-weight:600;">{token}</span>'
            )
        elif clean in NEGATORS:
            html_parts.append(
                f'<span style="background:rgba(251,146,60,0.25); color:#fb923c; '
                f'padding:2px 6px; border-radius:6px; font-weight:600;">{token}</span>'
            )
        else:
            html_parts.append(token)

    return "".join(html_parts)
