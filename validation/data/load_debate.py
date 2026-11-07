"""
Load the DEBATE dataset from HuggingFace raw CSVs (preserving topic from directory structure).

The HuggingFace flattened dataset loses topic information. We load directly from the raw
per-group CSV files cached by HuggingFace, where the topic is encoded in the directory name.

Directory structure: raw/{depth|breadth}/{topic_name}/{timestamp}_{topic}_{id}.csv
"""

import os
import glob
import pandas as pd


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "debate_events_with_topic.parquet")


def _find_raw_csvs() -> list:
    """Find raw CSV files in HuggingFace cache after dataset download."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--seantw--DEBATE_LLM")
    snapshots = glob.glob(os.path.join(hf_cache, "snapshots", "*"))
    if not snapshots:
        raise FileNotFoundError(
            "DEBATE dataset not found in HuggingFace cache. "
            "Run: from datasets import load_dataset; load_dataset('seantw/DEBATE_LLM')"
        )
    raw_dir = os.path.join(snapshots[0], "raw")
    csvs = glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return csvs


def _ensure_hf_download():
    """Download the dataset via HuggingFace if not already cached."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--seantw--DEBATE_LLM")
    if not os.path.exists(hf_cache):
        print("Downloading DEBATE dataset from HuggingFace...")
        from datasets import load_dataset
        load_dataset("seantw/DEBATE_LLM")
        print("Download complete.")


def load_debate_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load DEBATE dataset with topic column, caching as parquet.

    Returns deduplicated DataFrame with columns:
        topic, topic_category, group_file, chat_round_order, event_order,
        event_type, worker_id, sender_id, recipient_id, field, text,
        sliderValue, agreement_level, validity
    """
    if os.path.exists(CACHE_FILE) and not force_reload:
        print(f"Loading cached dataset from {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    _ensure_hf_download()
    csvs = _find_raw_csvs()
    print(f"Loading {len(csvs)} raw CSV files...")

    frames = []
    for csv_path in csvs:
        parts = csv_path.split(os.sep)
        # Structure: .../raw/{depth|breadth}/{topic_name}/{filename}.csv
        topic_name = os.path.basename(os.path.dirname(csv_path))
        # Category is depth or breadth
        raw_idx = parts.index("raw")
        topic_category = parts[raw_idx + 1] if raw_idx + 1 < len(parts) else "unknown"
        group_file = os.path.basename(csv_path)

        df = pd.read_csv(csv_path)
        df["topic"] = topic_name
        df["topic_category"] = topic_category
        df["group_file"] = group_file
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Ensure string columns are consistently typed (some CSVs have mixed types)
    str_cols = ["worker_id", "sender_id", "recipient_id", "field", "text",
                "agreement_level", "validity"]
    for col in str_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).replace("nan", pd.NA)

    # Deduplicate (the HF dataset has exact duplicate rows)
    before = len(combined)
    combined = combined.drop_duplicates()
    print(f"Deduplicated: {before} -> {len(combined)} rows")

    os.makedirs(CACHE_DIR, exist_ok=True)
    combined.to_parquet(CACHE_FILE, index=False)
    print(f"Cached {len(combined)} rows to {CACHE_FILE}")

    return combined
