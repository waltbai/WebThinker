"""Run solution experiments."""

import argparse
import json
import os
from datetime import datetime

import nltk

from src.webthinker.config import NLTK_DATA_PATH
from src.webthinker.graph import webthinker


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaia",
        choices=["gaia", ],
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        default=False
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = get_args()

    # Set paths
    nltk.data.path.append(NLTK_DATA_PATH)
    subdir = f"webthinker_{args.dataset}" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", subdir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Set LangSmith
    if args.langsmith:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "webthinker"

    # Init agent
    agent = webthinker()

    # Read tasks
    fp = os.path.join("data", "datasets", f"{args.dataset}.json")
    with open(fp, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Process task
    results = []
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
        solution = response.get("solution", "")
        results.append({
            "id": task["id"],
            "ground_truth": task["answer"],
            "model_output": solution,
        })
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
