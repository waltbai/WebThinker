"""Run solution experiments."""

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
import nltk

from webthinker.config import NLTK_DATA_PATH
from webthinker.evaluate import evaluate_qa, identify_group
from webthinker.graph import webthinker


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
        "--ids",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--llm_eval",
        action="store_true",
        default=False
    )
    return parser.parse_args()


def main():
    """Main function."""
    load_dotenv()
    args = get_args()

    # Set paths
    nltk.data.path.append(NLTK_DATA_PATH)
    subdir = f"webthinker_{args.dataset}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # Process tasks
    if args.ids != "all":
        selected_ids = [int(i) for i in args.ids.split(",")]
    else:
        selected_ids = [task["id"] for task in tasks]
    results = []
    for task in tasks:
        if task["id"] not in selected_ids:
            continue
        try:
            response = agent.invoke(
                {
                    "research_question": task["Question"],
                    "log_file": os.path.join(output_dir, f"{task['id']:0>2}.log"),
                },
                {"recursion_limit": 200}
            )
            solution = response.get("solution", "")
        except Exception:
            solution = ""

        results.append({
            "id": task["id"],
            "question": task["Question"],
            "label": task["answer"],
            "pred": solution,
            "group": identify_group(task),
        })

    # Evaluate results
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    performance = evaluate_qa(results, llm_eval=args.llm_eval)
    print("Performance:", performance)
    with open(os.path.join(output_dir, "performance.json"), "w", encoding="utf-8") as f:
        json.dump(performance, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
