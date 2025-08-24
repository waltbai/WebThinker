# WebThinker

This repository is a community version of [WebThinker](https://github.com/RUC-NLPIR/WebThinker).
It aims to explore how to convert the completion mode agent into chat mode.

*Notice*: This is not a perfect implementation of WebThinker,
thus the experimental results are **just for reference**.

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
| GAIA    | 25.48 | 36.89 | 19.42 |     35.92 |

Grouped:

| Dataset | Group |    F1 |   Acc |    EM | LLM Score |
| ------- | ----- | ----: | ----: | ----: | --------: |
| GAIA    | 1     | 37.17 | 48.72 | 28.21 |     43.59 |
| GAIA    | 2     | 21.72 | 30.77 | 17.31 |     36.54 |
| GAIA    | 3     |  3.77 | 25.00 |  0.00 |      8.33 |
