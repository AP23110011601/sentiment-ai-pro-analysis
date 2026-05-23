import pandas as pd
import re

# ── NOTE ──────────────────────────────────────────────────────────────────
# With tweet_eval dataset, labels are ALREADY 3-class:
#   0 = Negative,  1 = Neutral,  2 = Positive
# No synthetic neutral remapping needed anymore!
# ──────────────────────────────────────────────────────────────────────────


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the dataframe for training.

    Expects columns: 'text' and 'label' (0/1/2).
    Outputs columns: 'review_text' and 'label'.
    """
    df = df.copy()

    # Rename text column to match training pipeline
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'review_text'})

    # Basic text cleaning
    df['review_text'] = df['review_text'].astype(str)
    df['review_text'] = df['review_text'].apply(_clean_tweet)

    # Drop empty rows
    df = df[df['review_text'].str.strip().astype(bool)].reset_index(drop=True)

    print("3-class label distribution:\n", df['label'].value_counts().sort_index())
    return df


def _clean_tweet(text: str) -> str:
    """Clean tweet text — remove handles, URLs, extra whitespace."""
    text = re.sub(r"@\w+", "", text)           # Remove @mentions
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"#(\w+)", r"\1", text)       # Remove # but keep word
    text = re.sub(r"\s+", " ", text).strip()    # Collapse whitespace
    return text.lower()
