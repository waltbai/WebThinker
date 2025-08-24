# WebThinker

This repository is a community version of [WebThinker](https://github.com/RUC-NLPIR/WebThinker).
It aims to explore how to convert the completion mode agent into chat mode.

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

```shell
python -m src.webthinker.run --dataset gaia
```

### Generate Reports

```shell
python -m src.webthinker.run_report --dataset glaive
```

## Difference with official code

1. This version is based on [LangGraph](https://langchain-ai.github.io/langgraph/).
2. Since some LLM does not provide completion mode API,
we use the chat mode instead of the completion mode in this version.
3. Search query and report drafting tools are implemented in a standard tool calling way.
4. Some validation check are removed since they have never been triggered.
5. We do not implement the deep web explorer.
6. Prompts are slightly modified to adapt the chat mode and tool calling.
7. Currently, LangGraph store only supports vector indexing,
thus, we did not use the official store.
8. We add a summarize solution step in problem solving mode.
9. We support both google serper and tavily search (default to use tavily search).

## Preliminary Result

### Problem Solving Results

Overall:

| Dataset |    F1 |   Acc |    EM | LLM Score |
| ------- | ----: | ----: | ----: | --------: |
| GAIA    | 23.54 | 33.01 | 18.45 |     34.95 |

Grouped:

| Dataset | Group |    F1 |   Acc |    EM | LLM Score |
| ------- | ----- | ----: | ----: | ----: | --------: |
| GAIA    | 1     | 30.13 | 38.46 | 23.08 |     48.72 |
| GAIA    | 2     | 23.11 | 34.62 | 19.23 |     30.77 |
| GAIA    | 3     |  3.94 |  0.08 |  0.00 |      8.33 |
