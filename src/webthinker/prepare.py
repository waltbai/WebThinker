"""Prepare."""

import base64
import os

import nltk

from src.webthinker.config import NLTK_DATA_PATH


def main():
    """Main function."""
    # Init nltk
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_PATH)

    # Init crawl4ai
    os.system("crawl4ai-setup")
    os.system("crawl4ai-doctor")

    # Decode data
    datasets = ["gaia", "gpqa", "hle", "webwalkerqa"]
    for dataset in datasets:
        in_path = f"data/encoded/{dataset}.json"
        out_path = f"data/datasets/{dataset}.json"
        with open(in_path, "r", encoding="utf-8") as f:
            enc_data = f.read()
        dec_data = base64.b64decode(enc_data.encode("utf-8")).decode("utf-8")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dec_data)
        print(f"Dataset {dataset} decoded successfully.")


if __name__ == "__main__":
    main()
