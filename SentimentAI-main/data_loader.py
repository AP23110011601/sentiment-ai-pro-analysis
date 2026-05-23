from datasets import load_dataset
import pandas as pd


def load_data(max_samples=1000):
    """
    Load the tweet_eval sentiment dataset (real 3-class labels).
    Labels: 0 = Negative, 1 = Neutral, 2 = Positive

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to load (keeps training fast on CPU).
    """
    dataset = load_dataset("tweet_eval", "sentiment")

    # Combine train + validation for more data, then sample
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Rename columns to match existing pipeline
    df = df.rename(columns={"text": "text", "label": "label"})

    # Sample to keep training manageable on CPU
    if max_samples and len(df) > max_samples:
        # Stratified sample to keep class balance
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_samples // 3), random_state=42)
        ).reset_index(drop=True)

    print(f"Loaded {len(df)} samples from tweet_eval")
    print("Columns:", df.columns.tolist())
    print("Label distribution:\n", df['label'].value_counts().sort_index())
    print(df.head())

    return df
