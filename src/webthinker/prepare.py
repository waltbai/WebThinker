"""Prepare."""

import os

import nltk

from src.webthinker.config import NLTK_DATA_PATH


def main():
    """Main function."""
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_PATH)


if __name__ == "__main__":
    main()
