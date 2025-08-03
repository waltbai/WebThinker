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

### Generate Reports

```shell
python -m src.webthinker.run
```

## Difference with official code

1. This version is based on [LangGraph](https://langchain-ai.github.io/langgraph/).
2. Since some LLM does not provide completion mode API,
we use the chat mode instead of the completion mode in this version.
3. Search query and report drafting tools are implemented in a standard tool calling way.
4. Some validation check are removed since they have never been triggered.
5. We do not implement the deep web explorer.
6. Prompts are slightly modified to adapt the chat mode and tool calling.
7. Currently we only implement the report generation mode of webthinker.
