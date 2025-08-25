# WebThinker

This repository is a community version of [WebThinker](https://github.com/RUC-NLPIR/WebThinker).
It aims to learn the popular LangGraph framework and explore how to convert the completion mode agent into chat mode agent.

*Notice*: This is not a perfect implementation of WebThinker,
thus, the experimental results are **just for reference**.

## How to use

### Setup Environment

```shell
conda create -n webthinker python=3.12
```

### Install Requirements

```shell
pip install -r requirements.txt
```

### Prepare Data

```shell
python -m src.webthinker.prepare
```

### Solve Problem

Run inference and evaluation:

```shell
python -m src.webthinker.run --dataset gaia --ids all --llm_eval
```

Arguments:

- `--dataset`: specify the dataset.
- `--ids`: use "all" to run all samples or specify some IDs such as "1,2,3".
- `--langsmith`: whether to store intermediate steps in detail via LangSmith.
- `--llm_eval`: whether to use llm evaluation.

Only run evaluation:

```shell
python -m src.webthinker.evaluate --path /path/to/results.json --llm_eval
```

Arguments:

- `--path`: specify the result path.
- `--llm_eval`: whether to use llm evaluation.

### Generate Reports

```shell
python -m src.webthinker.run_report --dataset glaive
```

Arguments:

- `--dataset`: specify the dataset.
- `--ids`: use "all" to run all samples or specify some IDs such as "1,2,3".
- `--langsmith`: whether to store intermediate steps in detail via LangSmith.

## Difference with official code

1. This version is based on [LangGraph](https://langchain-ai.github.io/langgraph/).
2. Since some LLM does not provide completion mode API, we use the chat mode instead of the completion mode in this version. *This will potentially affect the reasoning coherence of the LLMs*.
3. Search query and report drafting tools are implemented in a standard tool calling way.
4. Some validation check are removed since they have never been triggered.
5. We do not implement the deep web explorer.
6. Prompts are slightly modified to adapt the chat mode and tool calling.
7. Currently, LangGraph store only supports vector indexing, thus, we did not use the official store.
8. We support both google serper and tavily search (default to use google search).

## Preliminary Result

### Problem Solving Results

Overall:

| Dataset |    F1 |   Acc |    EM | LLM Score |
| ------- | ----: | ----: | ----: | --------: |
| GAIA    | 46.83 | 41.75 | 25.24 |     40.78 |

Grouped:

| Dataset | Group |    F1 |   Acc |    EM | LLM Score |
| ------- | ----- | ----: | ----: | ----: | --------: |
| GAIA    | 1     | 59.01 | 43.59 | 30.77 |     48.72 |
| GAIA    | 2     | 41.35 | 44.23 | 25.00 |     30.77 |
| GAIA    | 3     | 30.98 | 25.00 |  8.33 |      8.33 |
