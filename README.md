# WebThinker

This repository is a community version of [WebThinker](https://github.com/RUC-NLPIR/WebThinker).
It aims to learn the popular [LangGraph](https://langchain-ai.github.io/langgraph/) framework and explore how to convert the completion mode agent into chat mode agent.

*Notice*: This is not a perfect implementation of WebThinker,
thus, the experimental results are **just for reference**.

## How to use

### Setup Environment

```shell
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
cp .env.example .env
```

### Install Requirements

```shell
uv sync
```

### Prepare Data

```shell
uv run prepare
```

*Notice*: Problem solving datasets in `data/encoded` are encoded to avoid web crawler.
They are decoded during data preparation.

### Solve Problem

Run inference and evaluation:

```shell
uv run qa --dataset gaia --ids all --llm_eval
```

Arguments:

- `--dataset`: specify the dataset.
- `--ids`: use "all" to run all samples or specify some IDs such as "1,2,3".
- `--langsmith`: whether to store intermediate steps in detail via [LangSmith](https://www.langchain.com/langsmith).
- `--llm_eval`: whether to use llm evaluation.

Only run evaluation:

```shell
uv run evaluate --path /path/to/results.json --llm_eval
```

Arguments:

- `--path`: specify the result path.
- `--llm_eval`: whether to use llm evaluation.

### Generate Reports

```shell
uv run report --dataset glaive --ids all
```

Arguments:

- `--dataset`: specify the dataset.
- `--ids`: use "all" to run all samples or specify some IDs such as "1,2,3".
- `--langsmith`: whether to store intermediate steps in detail via [LangSmith](https://www.langchain.com/langsmith).

## Difference with official code

1. This version is based on [LangGraph](https://langchain-ai.github.io/langgraph/).
2. Since some LLM does not provide completion mode API, we use the chat mode instead of the completion mode in this version. *This will potentially affect the reasoning coherence of the LLMs*.
3. Search query and report drafting tools are implemented in a standard tool calling way.
4. Some validation check are removed since they have never been triggered.
5. We do not implement the deep web explorer.
6. Prompts are slightly modified to adapt the chat mode and tool calling.
7. Currently, LangGraph store only supports vector indexing, thus, we did not use the official store.
8. We support both [google serper](https://serper.dev/) and [tavily](https://www.tavily.com/) search (default to use google serper).
9. We have not implemented the report evaluation.

## Preliminary Result

### Problem Solving Results

*Notice*: This is only the results for one test round, thus there may exist randomness.

Overall:

| Dataset |    F1 |   Acc |    EM | LLM Score |
| ------- | ----: | ----: | ----: | --------: |
| GAIA    | 46.83 | 41.75 | 25.24 |     40.78 |

Grouped:

| Dataset | Group |    F1 |   Acc |    EM | LLM Score |
| ------- | ----- | ----: | ----: | ----: | --------: |
| GAIA    | 1     | 59.01 | 43.59 | 30.77 |     46.15 |
| GAIA    | 2     | 41.35 | 44.23 | 25.00 |     43.59 |
| GAIA    | 3     | 30.98 | 25.00 |  8.33 |      8.33 |
