# WebThinker

This repository is a community version of [WebThinker](https://github.com/RUC-NLPIR/WebThinker).

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

| Dataset |    F1 |   Acc |    EM |
| ------- | ----: | ----: | ----: |
| GAIA    | 34.97 | 35.92 | 30.10 |

Grouped:

| Dataset | Group |    F1 |   Acc |    EM |
| ------- | ----- | ----: | ----: | ----: |
| GAIA    | 1     | 45.66 | 41.03 | 38.46 |
| GAIA    | 2     | 31.24 | 32.69 | 28.85 |
| GAIA    | 3     | 16.39 | 33.33 |  8.33 |
