"""Prepare."""

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


if __name__ == "__main__":
    main()
