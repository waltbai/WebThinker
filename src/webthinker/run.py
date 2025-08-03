"""Run experiments."""

import argparse
from datetime import datetime
import json
import os
import nltk

from src.webthinker.config import NLTK_DATA_PATH
from src.webthinker.graph_report import webthinker_report


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--langsmith", action="store_true", default=False)
    return parser.parse_args()


def main():
    """Main function."""
    args = get_args()

    # Set paths
    nltk.data.path.append(NLTK_DATA_PATH)
    subdir = "webthinker_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Set LangSmith
    if args.langsmith:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "webthinker"

    # Init agent
    agent = webthinker_report()

    # Read tasks
    with open("data/datasets/tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Process task
    for task in tasks:
        if task["id"] != 1:
            continue
        response = agent.invoke(
            {
                "research_question": task["Question"],
                "log_file": os.path.join(output_dir, f"{task['id']:0>2}.log"),
            },
            {"recursion_limit": 100}
        )
        article = response.get("article", "")
        fp = os.path.join(output_dir, f"{task['id']:0>2}.md")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(article)


if __name__ == "__main__":
    main()
